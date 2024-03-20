# Standard
from pathlib import Path
from typing import List
import shutil

# Third Party
import git


class MockTaxonomy:
    INIT_COMMIT_FILE = "README.md"

    def __init__(self, path: Path) -> None:
        self.root = path
        self._repo = git.Repo.init(path, initial_branch="main")
        with open(path / self.INIT_COMMIT_FILE, "wb"):
            pass
        self._repo.index.add([self.INIT_COMMIT_FILE])
        self._repo.index.commit("Initial commit")

    @property
    def untracked_files(self) -> List[str]:
        """List untracked files in the repository"""
        return self._repo.untracked_files

    def create_untracked(self, rel_path: str) -> None:
        """Create a new untracked file in the repository.

        Args:
            rel_path (str): Relative path (from repository root) to the file.
        """
        assert not Path(rel_path).is_absolute()
        file_path = Path(self.root / rel_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text("new taxonomy", encoding="utf-8")

    def add_tracked(self, rel_path: str) -> None:
        """Add a new tracked file to the repository (and commits it).

        Args:
            rel_path (str): Relative path (from repository root) to the file.
        """
        self.create_untracked(rel_path)
        self._repo.index.add([rel_path])
        self._repo.index.commit("new commit")

    def remove_file(self, rel_path: str) -> None:
        """Remove a file in the repository (tracked or not)

        Args:
            rel_path (str): Relative path (from repository root) to the file.
        """
        Path(self.root / rel_path).unlink(missing_ok=True)

    def teardown(self) -> None:
        """Recursively remove the temporary repository and all of its
        subdirectories and files.
        """
        shutil.rmtree(self.root)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.teardown()