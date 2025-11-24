import ast
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Set

class CodeEntity:
    def __init__(self, name: str, type: str, file_path: str, line_number: int, docstring: str = None, parent: str = None):
        self.name = name
        self.type = type
        self.file_path = file_path
        self.line_number = line_number
        self.docstring = docstring
        self.parent = parent
        self.dependencies: Set[str] = set()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "docstring": self.docstring,
            "parent": self.parent,
            "dependencies": list(self.dependencies)
        }

class KnowledgeGraphBuilder(ast.NodeVisitor):
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.entities: Dict[str, CodeEntity] = {}
        self.current_file = ""
        self.current_class = None
        self.imports: Dict[str, List[str]] = {} # file_path -> list of imported module names

    def analyze(self):
        for file_path in self.project_root.rglob("*.py"):
            if "venv" in str(file_path) or "__pycache__" in str(file_path):
                continue
            
            self.current_file = str(file_path.relative_to(self.project_root)).replace("\\", "/")
            self.imports[self.current_file] = []
            
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read(), filename=str(file_path))
                    self.visit(tree)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

        self._resolve_dependencies()

    def visit_ClassDef(self, node):
        entity_name = node.name
        full_name = f"{self.current_file}::{entity_name}"
        
        entity = CodeEntity(
            name=full_name,
            type="class",
            file_path=self.current_file,
            line_number=node.lineno,
            docstring=ast.get_docstring(node)
        )
        self.entities[full_name] = entity
        
        old_class = self.current_class
        self.current_class = full_name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        entity_name = node.name
        if self.current_class:
            full_name = f"{self.current_class}.{entity_name}"
            type_ = "method"
            parent = self.current_class
        else:
            full_name = f"{self.current_file}::{entity_name}"
            type_ = "function"
            parent = None

        entity = CodeEntity(
            name=full_name,
            type=type_,
            file_path=self.current_file,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            parent=parent
        )
        self.entities[full_name] = entity
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports[self.current_file].append(alias.name)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports[self.current_file].append(node.module)

    def _resolve_dependencies(self):
        # Simple resolution: map imports to files
        # This is a heuristic and won't be perfect without full python path resolution
        
        # Create a map of module name -> file path
        module_map = {}
        for file_path in self.imports.keys():
            # src/utils/gl_context.py -> src.utils.gl_context
            parts = file_path.replace(".py", "").split("/")
            module_name = ".".join(parts)
            module_map[module_name] = file_path
            
            # Also handle package imports (src/utils/__init__.py -> src.utils)
            if parts[-1] == "__init__":
                package_name = ".".join(parts[:-1])
                module_map[package_name] = file_path

        # Link entities based on imports
        for entity_name, entity in self.entities.items():
            file_imports = self.imports.get(entity.file_path, [])
            for imp in file_imports:
                # Try to find which file this import refers to
                # Check exact match
                target_file = module_map.get(imp)
                if not target_file:
                    # Check relative imports or partial matches
                    # (Simplified logic)
                    pass
                
                if target_file:
                    # Add dependency to the file (represented by a module node if we had one, 
                    # but here we just link to entities in that file? 
                    # Or just store the file dependency)
                    entity.dependencies.add(target_file)

    def export_json(self, output_path: str):
        data = {
            "entities": [e.to_dict() for e in self.entities.values()],
            "files": list(self.imports.keys()),
            "file_dependencies": self.imports
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def export_mermaid(self, output_path: str):
        lines = ["graph TD"]
        
        # Group by file (subgraphs)
        files = {}
        for name, entity in self.entities.items():
            if entity.file_path not in files:
                files[entity.file_path] = []
            files[entity.file_path].append(entity)

        # Create subgraphs
        for file_path, entities in files.items():
            clean_name = file_path.replace("/", "_").replace(".", "_").replace("-", "_")
            lines.append(f"    subgraph {clean_name} [{file_path}]")
            for entity in entities:
                # Create a node ID
                node_id = str(hash(entity.name)).replace("-", "N")
                # Label
                short_name = entity.name.split("::")[-1].split(".")[-1]
                if entity.type == "class":
                    shape = f"[{short_name}]"
                elif entity.type == "method":
                    shape = f"({short_name})"
                else:
                    shape = f"([{short_name}])"
                
                lines.append(f"        {node_id}{shape}")
                
                # Link to parent
                if entity.parent:
                    parent_id = str(hash(entity.parent)).replace("-", "N")
                    lines.append(f"        {parent_id} --> {node_id}")
            lines.append("    end")

        # Add file-level dependencies (simplified edges)
        # To avoid clutter, we'll just link the subgraphs or representative nodes?
        # For now, let's just link files based on imports to keep it readable
        # Actually, let's make a separate high-level graph for files
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def export_mermaid_class_diagram(self, output_path: str):
        lines = ["classDiagram"]
        
        for name, entity in self.entities.items():
            if entity.type == "class":
                class_name = entity.name.split("::")[-1]
                lines.append(f"    class {class_name}")
                
                # Find methods for this class
                methods = [e for e in self.entities.values() if e.parent == entity.name]
                for m in methods:
                    method_name = m.name.split(".")[-1]
                    lines.append(f"    {class_name} : +{method_name}()")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

if __name__ == "__main__":
    project_root = "d:/I_Drive_Backup/Projects/game_design/map_gen"
    builder = KnowledgeGraphBuilder(project_root)
    builder.analyze()
    
    os.makedirs("docs/architecture", exist_ok=True)
    builder.export_json("docs/architecture/knowledge_graph.json")
    builder.export_mermaid_class_diagram("docs/architecture/knowledge_graph_classes.md")
    print("Knowledge graph generated.")
