import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import pickle
import tempfile
import shutil
from pathlib import Path
import numpy as np

from euclid.rag import (
    Document, 
    DocumentChunk, 
    Collection, 
    VectorDB,
    create_collection,
    list_collections,
    add_document,
    query_collection,
    delete_collection,
    HAVE_SENTENCE_TRANSFORMERS
)


class TestDocumentModels(unittest.TestCase):
    def test_document_creation(self):
        doc = Document(content="Test content")
        self.assertEqual(doc.content, "Test content")
        self.assertIsNotNone(doc.id)
        self.assertEqual(doc.metadata, {})
        self.assertIsNone(doc.embedding)
        
        # Test with metadata and embedding
        doc = Document(
            content="Test with metadata",
            metadata={"title": "Test", "source": "Unit test"},
            embedding=[0.1, 0.2, 0.3]
        )
        self.assertEqual(doc.content, "Test with metadata")
        self.assertEqual(doc.metadata["title"], "Test")
        self.assertEqual(doc.embedding, [0.1, 0.2, 0.3])
    
    def test_document_chunk_creation(self):
        chunk = DocumentChunk(
            content="Chunk content",
            document_id="doc123"
        )
        self.assertEqual(chunk.content, "Chunk content")
        self.assertEqual(chunk.document_id, "doc123")
        self.assertIsNotNone(chunk.id)
        
        # Test with metadata and embedding
        chunk = DocumentChunk(
            content="Chunk with metadata",
            metadata={"section": "intro"},
            embedding=[0.4, 0.5, 0.6],
            document_id="doc456"
        )
        self.assertEqual(chunk.content, "Chunk with metadata")
        self.assertEqual(chunk.metadata["section"], "intro")
        self.assertEqual(chunk.embedding, [0.4, 0.5, 0.6])
        self.assertEqual(chunk.document_id, "doc456")
    
    def test_collection_creation(self):
        collection = Collection(id="col123", name="Test Collection")
        self.assertEqual(collection.id, "col123")
        self.assertEqual(collection.name, "Test Collection")
        self.assertIsNone(collection.description)
        self.assertEqual(collection.documents, {})
        self.assertEqual(collection.chunks, {})
        self.assertEqual(collection.metadata, {})
        
        # Test with description
        collection = Collection(
            id="col456",
            name="Another Collection",
            description="Test description"
        )
        self.assertEqual(collection.description, "Test description")


@patch('euclid.rag.HAVE_SENTENCE_TRANSFORMERS', True)
class TestVectorDBWithEmbeddings(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db = VectorDB(db_path=self.temp_dir)
        
        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        self.db.embedding_model = mock_model
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_create_collection(self, mock_json_dump, mock_file_open):
        collection_id = self.db.create_collection("Test Collection", "Test Description")
        
        self.assertIsNotNone(collection_id)
        mock_file_open.assert_called()
        mock_json_dump.assert_called()
        
        # Verify collection was saved
        mock_pickle_dump = mock_open()
        with patch('builtins.open', mock_pickle_dump):
            with patch('pickle.dump') as mock_dump:
                self.db._save_collection(Collection(id=collection_id, name="Test"))
                mock_dump.assert_called_once()
    
    @patch('pathlib.Path.glob')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_list_collections(self, mock_json_load, mock_file_open, mock_glob):
        # Mock file paths
        mock_path1 = MagicMock()
        mock_path1.stem = "col1"
        mock_path2 = MagicMock()
        mock_path2.stem = "col2"
        mock_glob.return_value = [mock_path1, mock_path2]
        
        # Mock JSON data
        mock_json_load.side_effect = [
            {"id": "col1", "name": "Collection 1", "description": "Desc 1", "doc_count": 5, "chunk_count": 20},
            {"id": "col2", "name": "Collection 2", "description": "Desc 2", "doc_count": 3, "chunk_count": 15}
        ]
        
        collections = self.db.list_collections()
        
        self.assertEqual(len(collections), 2)
        self.assertEqual(collections[0]["id"], "col1")
        self.assertEqual(collections[0]["name"], "Collection 1")
        self.assertEqual(collections[0]["doc_count"], 5)
        self.assertEqual(collections[1]["id"], "col2")
        self.assertEqual(collections[1]["chunk_count"], 15)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_get_collection(self, mock_pickle_load, mock_file_open):
        # Mock collection
        mock_collection = Collection(id="test_col", name="Test")
        mock_pickle_load.return_value = mock_collection
        
        # Mock file exists
        with patch('pathlib.Path.exists', return_value=True):
            collection = self.db.get_collection("test_col")
            
            self.assertEqual(collection.id, "test_col")
            self.assertEqual(collection.name, "Test")
            mock_file_open.assert_called_once()
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.unlink')
    def test_delete_collection(self, mock_unlink, mock_exists):
        mock_exists.return_value = True
        
        result = self.db.delete_collection("test_col")
        
        self.assertTrue(result)
        self.assertEqual(mock_unlink.call_count, 2)  # PKL and JSON files
        
        # Test non-existent collection
        mock_exists.return_value = False
        result = self.db.delete_collection("nonexistent")
        self.assertFalse(result)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('pickle.dump')
    def test_add_document(self, mock_pickle_dump, mock_pickle_load, mock_file_open):
        # Mock collection
        mock_collection = Collection(id="test_col", name="Test")
        mock_pickle_load.return_value = mock_collection
        
        # Mock file exists
        with patch('pathlib.Path.exists', return_value=True):
            doc_id = self.db.add_document(
                "test_col", 
                "Test document content",
                {"title": "Test Doc"}
            )
            
            self.assertIsNotNone(doc_id)
            # Verify document was added
            self.assertIn(doc_id, mock_collection.documents)
            # Verify document was chunked
            self.assertGreater(len(mock_collection.chunks), 0)
            # Verify encoding was called
            self.db.embedding_model.encode.assert_called()
            # Verify collection was saved
            mock_pickle_dump.assert_called()
    
    def test_chunk_document(self):
        doc = Document(
            id="doc123",
            content="This is a test document with multiple sentences. It should be split into chunks. "
                   "Here is another sentence. And another one. Let's make this long enough to create multiple chunks."
        )
        
        chunks = self.db._chunk_document(doc)
        
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.document_id, "doc123")
            self.assertIsNotNone(chunk.content)
            # Verify embedding was created
            self.assertIsNotNone(chunk.embedding)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_search_with_embeddings(self, mock_pickle_load, mock_file_open):
        # Create mock collection with documents and chunks
        collection = Collection(id="test_col", name="Test")
        
        # Add documents with embeddings
        doc1 = Document(
            id="doc1",
            content="This is the first test document",
            embedding=[0.9, 0.1, 0.1]  # More similar to query
        )
        doc2 = Document(
            id="doc2",
            content="This is the second test document",
            embedding=[0.1, 0.9, 0.1]  # Less similar to query
        )
        
        # Add chunks with embeddings
        chunk1 = DocumentChunk(
            id="chunk1",
            content="This is the first chunk",
            document_id="doc1",
            embedding=[0.8, 0.1, 0.2]  # More similar to query
        )
        chunk2 = DocumentChunk(
            id="chunk2",
            content="This is the second chunk",
            document_id="doc2",
            embedding=[0.2, 0.8, 0.2]  # Less similar to query
        )
        
        # Add to collection
        collection.documents["doc1"] = doc1
        collection.documents["doc2"] = doc2
        collection.chunks["chunk1"] = chunk1
        collection.chunks["chunk2"] = chunk2
        
        mock_pickle_load.return_value = collection
        
        # Set up mock for embedding model
        self.db.embedding_model.encode.return_value = np.array([0.9, 0.1, 0.1])
        
        # Mock file exists
        with patch('pathlib.Path.exists', return_value=True):
            # Search documents
            results = self.db.search("test_col", "test query", use_chunks=False)
            
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["id"], "doc1")  # Most similar should be first
            self.assertGreater(results[0]["similarity"], results[1]["similarity"])
            
            # Search chunks
            results = self.db.search("test_col", "test query", use_chunks=True)
            
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["id"], "chunk1")  # Most similar should be first
            self.assertGreater(results[0]["similarity"], results[1]["similarity"])
    
    def test_cosine_similarity(self):
        # Test with identical vectors
        vec1 = [0.1, 0.2, 0.3]
        vec2 = [0.1, 0.2, 0.3]
        similarity = self.db._cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0)
        
        # Test with orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = self.db._cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.0)
        
        # Test with somewhat similar vectors
        vec1 = [0.8, 0.1, 0.1]
        vec2 = [0.7, 0.2, 0.1]
        similarity = self.db._cosine_similarity(vec1, vec2)
        self.assertGreater(similarity, 0.9)  # Should be highly similar


@patch('euclid.rag.HAVE_SENTENCE_TRANSFORMERS', False)
class TestVectorDBWithoutEmbeddings(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db = VectorDB(db_path=self.temp_dir)
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_keyword_search(self, mock_pickle_load, mock_file_open):
        # Create mock collection with documents
        collection = Collection(id="test_col", name="Test")
        
        # Add documents without embeddings
        doc1 = Document(
            id="doc1",
            content="This document mentions keywords like python and machine learning"
        )
        doc2 = Document(
            id="doc2",
            content="This document is about something else entirely"
        )
        
        # Add to collection
        collection.documents["doc1"] = doc1
        collection.documents["doc2"] = doc2
        
        mock_pickle_load.return_value = collection
        
        # Mock file exists
        with patch('pathlib.Path.exists', return_value=True):
            # Search documents with keyword matching
            results = self.db.search("test_col", "python machine learning", use_chunks=False)
            
            self.assertEqual(len(results), 2)  # Both documents match (at least partially)
            self.assertEqual(results[0]["id"], "doc1")  # Most relevant should be first
            self.assertGreater(results[0]["similarity"], results[1]["similarity"])


class TestRAGFunctions(unittest.TestCase):
    @patch('euclid.rag.vectordb')
    def test_create_collection_function(self, mock_vectordb):
        # Mock the underlying VectorDB method
        mock_vectordb.create_collection.return_value = "test-col-id"
        
        result = create_collection("Test Collection", "Test Description")
        
        mock_vectordb.create_collection.assert_called_once_with("Test Collection", "Test Description")
        self.assertIn("Collection created", result)
        self.assertIn("test-col-id", result)
    
    @patch('euclid.rag.vectordb')
    def test_list_collections_function_with_collections(self, mock_vectordb):
        # Mock the underlying VectorDB method
        mock_vectordb.list_collections.return_value = [
            {
                "id": "col1",
                "name": "Collection 1",
                "description": "Description 1",
                "doc_count": 5,
                "chunk_count": 10
            },
            {
                "id": "col2",
                "name": "Collection 2",
                "description": None,
                "doc_count": 3,
                "chunk_count": 6
            }
        ]
        
        result = list_collections()
        
        mock_vectordb.list_collections.assert_called_once()
        self.assertIn("# Available Collections", result)
        self.assertIn("Collection 1", result)
        self.assertIn("Collection 2", result)
        self.assertIn("Description 1", result)
        self.assertIn("Documents: 5", result)
    
    @patch('euclid.rag.vectordb')
    def test_list_collections_function_empty(self, mock_vectordb):
        # Mock empty collection list
        mock_vectordb.list_collections.return_value = []
        
        result = list_collections()
        
        self.assertIn("No collections found", result)
    
    @patch('euclid.rag.vectordb')
    def test_add_document_function(self, mock_vectordb):
        # Mock successful document addition
        mock_vectordb.add_document.return_value = "doc-id-123"
        
        result = add_document(
            "test-col-id",
            "Document content",
            title="Test Document",
            source="Unit Test"
        )
        
        self.assertIn("Document added", result)
        self.assertIn("doc-id-123", result)
        
        # Check metadata was passed correctly
        call_args = mock_vectordb.add_document.call_args
        self.assertEqual(call_args[0][0], "test-col-id")
        self.assertEqual(call_args[0][1], "Document content")
        self.assertEqual(call_args[0][2]["title"], "Test Document")
        self.assertEqual(call_args[0][2]["source"], "Unit Test")
    
    @patch('euclid.rag.vectordb')
    def test_add_document_function_failed(self, mock_vectordb):
        # Mock failed document addition
        mock_vectordb.add_document.return_value = None
        
        result = add_document("nonexistent-col", "Content")
        
        self.assertIn("Error", result)
        self.assertIn("not found", result)
    
    @patch('euclid.rag.vectordb')
    def test_query_collection_function(self, mock_vectordb):
        # Mock search results
        mock_vectordb.search.return_value = [
            {
                "id": "chunk1",
                "content": "This is chunk 1 content",
                "similarity": 0.95,
                "metadata": {"title": "Doc 1", "source": "Source 1"},
                "document_id": "doc1"
            },
            {
                "id": "chunk2",
                "content": "This is chunk 2 content",
                "similarity": 0.85,
                "metadata": {"title": "Doc 2"},
                "document_id": "doc2"
            }
        ]
        
        result = query_collection("test-col-id", "test query", top_k=2)
        
        mock_vectordb.search.assert_called_once_with("test-col-id", "test query", 2, True)
        self.assertIn("# Search Results for: 'test query'", result)
        self.assertIn("Result 1 (Relevance: 0.95)", result)
        self.assertIn("Result 2 (Relevance: 0.85)", result)
        self.assertIn("This is chunk 1 content", result)
        self.assertIn("This is chunk 2 content", result)
        self.assertIn("Doc 1", result)
        self.assertIn("Source 1", result)
    
    @patch('euclid.rag.vectordb')
    def test_query_collection_function_no_results(self, mock_vectordb):
        # Mock empty search results
        mock_vectordb.search.return_value = []
        
        result = query_collection("test-col-id", "test query")
        
        self.assertIn("No results found", result)
    
    @patch('euclid.rag.vectordb')
    def test_delete_collection_function_success(self, mock_vectordb):
        # Mock successful deletion
        mock_vectordb.delete_collection.return_value = True
        
        result = delete_collection("test-col-id")
        
        mock_vectordb.delete_collection.assert_called_once_with("test-col-id")
        self.assertIn("Collection deleted", result)
    
    @patch('euclid.rag.vectordb')
    def test_delete_collection_function_failure(self, mock_vectordb):
        # Mock failed deletion
        mock_vectordb.delete_collection.return_value = False
        
        result = delete_collection("nonexistent-col")
        
        self.assertIn("Error", result)
        self.assertIn("not found", result)


if __name__ == '__main__':
    unittest.main()