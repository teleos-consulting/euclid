import json
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

from euclid.conversation import Conversation
from euclid.ollama import Message


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.conversation = Conversation(id="test-conversation")
    
    def test_add_message(self):
        self.conversation.add_message("user", "Hello")
        self.assertEqual(len(self.conversation.messages), 1)
        self.assertEqual(self.conversation.messages[0].role, "user")
        self.assertEqual(self.conversation.messages[0].content, "Hello")
        
        self.conversation.add_message("assistant", "Hi there")
        self.assertEqual(len(self.conversation.messages), 2)
        self.assertEqual(self.conversation.messages[1].role, "assistant")
        self.assertEqual(self.conversation.messages[1].content, "Hi there")
    
    def test_formatted_messages(self):
        self.conversation.add_message("user", "Hello")
        self.conversation.add_message("assistant", "Hi there")
        
        formatted = self.conversation.formatted_messages
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["role"], "user")
        self.assertEqual(formatted[0]["content"], "Hello")
        self.assertEqual(formatted[1]["role"], "assistant")
        self.assertEqual(formatted[1]["content"], "Hi there")
    
    @patch('euclid.conversation.config')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_save_new_conversation(self, mock_mkdir, mock_exists, mock_file_open, mock_config):
        mock_config.history_file = Path("/tmp/test_history.json")
        mock_config.max_history_length = 10
        mock_exists.return_value = False
        
        self.conversation.add_message("user", "Hello")
        self.conversation.save()
        
        mock_mkdir.assert_called_once()
        mock_file_open.assert_called_once_with(mock_config.history_file, "w")
        
        handle = mock_file_open()
        file_content = handle.write.call_args[0][0]
        parsed_content = json.loads(file_content)
        
        self.assertEqual(len(parsed_content), 1)
        self.assertEqual(parsed_content[0]["id"], "test-conversation")
        self.assertEqual(len(parsed_content[0]["messages"]), 1)
    
    @patch('euclid.conversation.config')
    @patch('builtins.open', new_callable=mock_open, read_data='[{"id": "test-conversation", "messages": []}]')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_save_existing_conversation(self, mock_mkdir, mock_exists, mock_file_open, mock_config):
        mock_config.history_file = Path("/tmp/test_history.json")
        mock_config.max_history_length = 10
        mock_exists.return_value = True
        
        self.conversation.add_message("user", "Updated message")
        self.conversation.save()
        
        mock_mkdir.assert_called_once()
        handle = mock_file_open()
        file_content = handle.write.call_args[0][0]
        parsed_content = json.loads(file_content)
        
        self.assertEqual(len(parsed_content), 1)
        self.assertEqual(parsed_content[0]["id"], "test-conversation")
        self.assertEqual(len(parsed_content[0]["messages"]), 1)
    
    @patch('euclid.conversation.config')
    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    @patch('pathlib.Path.exists')
    def test_save_with_corrupted_file(self, mock_exists, mock_file_open, mock_config):
        mock_config.history_file = Path("/tmp/test_history.json")
        mock_config.max_history_length = 10
        mock_exists.return_value = True
        
        self.conversation.add_message("user", "Hello")
        self.conversation.save()
        
        handle = mock_file_open()
        file_content = handle.write.call_args[0][0]
        parsed_content = json.loads(file_content)
        
        self.assertEqual(len(parsed_content), 1)
        self.assertEqual(parsed_content[0]["id"], "test-conversation")
    
    @patch('euclid.conversation.config')
    @patch('builtins.open', new_callable=mock_open, read_data='[{"id": "test-conversation", "messages": [{"role": "user", "content": "Hello"}]}]')
    @patch('pathlib.Path.exists')
    def test_load_existing_conversation(self, mock_exists, mock_file_open, mock_config):
        mock_config.history_file = Path("/tmp/test_history.json")
        mock_exists.return_value = True
        
        conversation = Conversation.load("test-conversation")
        
        self.assertEqual(conversation.id, "test-conversation")
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].role, "user")
        self.assertEqual(conversation.messages[0].content, "Hello")
    
    @patch('euclid.conversation.config')
    @patch('pathlib.Path.exists')
    def test_load_nonexistent_conversation(self, mock_exists, mock_config):
        mock_config.history_file = Path("/tmp/test_history.json")
        mock_exists.return_value = False
        
        conversation = Conversation.load("nonexistent-conversation")
        
        self.assertEqual(conversation.id, "nonexistent-conversation")
        self.assertEqual(len(conversation.messages), 0)
    
    @patch('euclid.conversation.config')
    @patch('builtins.open', new_callable=mock_open, read_data='[{"id": "conv1"}, {"id": "conv2"}]')
    @patch('pathlib.Path.exists')
    def test_list_conversations(self, mock_exists, mock_file_open, mock_config):
        mock_config.history_file = Path("/tmp/test_history.json")
        mock_exists.return_value = True
        
        conversations = Conversation.list_conversations()
        
        self.assertEqual(len(conversations), 2)
        self.assertEqual(conversations[0]["id"], "conv1")
        self.assertEqual(conversations[1]["id"], "conv2")
    
    @patch('euclid.conversation.config')
    @patch('pathlib.Path.exists')
    def test_list_conversations_no_file(self, mock_exists, mock_config):
        mock_config.history_file = Path("/tmp/test_history.json")
        mock_exists.return_value = False
        
        conversations = Conversation.list_conversations()
        
        self.assertEqual(conversations, [])


if __name__ == '__main__':
    unittest.main()