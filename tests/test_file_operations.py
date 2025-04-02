import unittest
import tempfile
import os
from pathlib import Path

from euclid.tools.file_operations import view_file, glob_tool, ls_tool, edit_file, replace_file


class TestFileOperations(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create some test files
        self.test_file1 = self.test_dir / "test1.txt"
        self.test_file1.write_text("This is test file 1\nWith multiple lines\nFor testing purposes")
        
        self.test_file2 = self.test_dir / "test2.py"
        self.test_file2.write_text("def test_function():\n    return 'This is a test'")
        
        # Create a subdirectory
        self.sub_dir = self.test_dir / "subdir"
        self.sub_dir.mkdir()
        
        self.test_file3 = self.sub_dir / "test3.txt"
        self.test_file3.write_text("This is test file 3 in the subdirectory")
    
    def tearDown(self):
        # Remove temporary directory and files
        self.temp_dir.cleanup()
    
    def test_view_file(self):
        """Test viewing a file."""
        # Test normal text file
        result = view_file(str(self.test_file1))
        self.assertIn("This is test file 1", result)
        self.assertIn("With multiple lines", result)
        
        # Test with offset and limit
        result = view_file(str(self.test_file1), offset=1, limit=1)
        self.assertNotIn("This is test file 1", result)  # Skipped by offset
        self.assertIn("With multiple lines", result)
        self.assertNotIn("For testing purposes", result)  # Limited
        
        # Test non-existent file
        result = view_file(str(self.test_dir / "nonexistent.txt"))
        self.assertIn("Error", result)
    
    def test_glob_tool(self):
        """Test glob tool."""
        # Find all text files
        result = glob_tool("*.txt", str(self.test_dir))
        self.assertIn("test1.txt", result)
        self.assertNotIn("test2.py", result)  # Not a txt file
        
        # Recursive search
        result = glob_tool("**/*.txt", str(self.test_dir))
        self.assertIn("test1.txt", result)
        self.assertIn("test3.txt", result)  # In subdirectory
        
        # Non-matching pattern
        result = glob_tool("*.jpg", str(self.test_dir))
        self.assertIn("No files matching", result)
    
    def test_ls_tool(self):
        """Test listing directory contents."""
        # List root test directory
        result = ls_tool(str(self.test_dir))
        self.assertIn("test1.txt", result)
        self.assertIn("test2.py", result)
        self.assertIn("subdir", result)
        
        # List with ignore pattern
        result = ls_tool(str(self.test_dir), ignore=["*.py"])
        self.assertIn("test1.txt", result)
        self.assertNotIn("test2.py", result)  # Ignored
        
        # Test non-existent directory
        result = ls_tool(str(self.test_dir / "nonexistent"))
        self.assertIn("Error", result)
    
    def test_edit_file(self):
        """Test editing a file."""
        # Test modifying existing content
        result = edit_file(
            str(self.test_file1),
            "This is test file 1\n",
            "This is MODIFIED test file 1\n"
        )
        self.assertIn("updated", result)
        
        # Verify the file was updated
        content = self.test_file1.read_text()
        self.assertIn("MODIFIED", content)
        
        # Test creating a new file
        new_file = self.test_dir / "new_file.txt"
        result = edit_file(str(new_file), "", "This is a new file")
        self.assertIn("created", result)
        
        # Verify the file was created
        self.assertTrue(new_file.exists())
        self.assertEqual(new_file.read_text(), "This is a new file")
    
    def test_replace_file(self):
        """Test replacing a file."""
        # Test replacing an existing file
        result = replace_file(str(self.test_file1), "Complete replacement content")
        self.assertIn("successfully", result)
        
        # Verify the file was replaced
        content = self.test_file1.read_text()
        self.assertEqual(content, "Complete replacement content")
        
        # Test creating a new file
        new_file = self.test_dir / "replaced_new.txt"
        result = replace_file(str(new_file), "New file content")
        self.assertIn("successfully", result)
        
        # Verify the file was created
        self.assertTrue(new_file.exists())
        self.assertEqual(new_file.read_text(), "New file content")


if __name__ == "__main__":
    unittest.main()
