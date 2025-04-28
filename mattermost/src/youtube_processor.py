from typing import List, Dict, Any, Optional, Tuple
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import re
from datetime import timedelta

logger = logging.getLogger(__name__)

class YouTubeProcessor:
    """Handles YouTube video processing, including transcript fetching and timestamp identification."""
    
    def __init__(self):
        self.transcript_api = YouTubeTranscriptApi
        self.formatter = TextFormatter()
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract YouTube video ID from various URL formats.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID if found, None otherwise
        """
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Regular YouTube URLs
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',  # Shortened URLs
            r'^([0-9A-Za-z_-]{11})$'  # Direct video ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Fetch transcript for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of transcript segments with timestamps
        """
        try:
            transcript = self.transcript_api.get_transcript(video_id)
            return transcript
        except Exception as e:
            logger.error(f"Error fetching transcript for video {video_id}: {e}")
            return []
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to HH:MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        return str(timedelta(seconds=int(seconds)))
    
    def find_relevant_segments(
        self,
        transcript: List[Dict[str, Any]],
        query: str,
        context_window: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find transcript segments relevant to the query.
        
        Args:
            transcript: List of transcript segments
            query: Search query
            context_window: Number of segments to include before and after matches
            
        Returns:
            List of relevant segments with timestamps
        """
        query_terms = query.lower().split()
        relevant_segments = []
        
        for i, segment in enumerate(transcript):
            text = segment['text'].lower()
            
            # Check if any query term appears in the segment
            if any(term in text for term in query_terms):
                # Get context window
                start_idx = max(0, i - context_window)
                end_idx = min(len(transcript), i + context_window + 1)
                
                # Add segments with context
                context_segments = transcript[start_idx:end_idx]
                
                # Format segments with timestamps
                formatted_segment = {
                    'timestamp': self.format_timestamp(segment['start']),
                    'text': ' '.join([s['text'] for s in context_segments]),
                    'start_time': segment['start'],
                    'url_time': f"&t={int(segment['start'])}s"
                }
                
                relevant_segments.append(formatted_segment)
        
        # Sort by timestamp and remove duplicates
        relevant_segments.sort(key=lambda x: x['start_time'])
        return self._remove_overlapping_segments(relevant_segments)
    
    def _remove_overlapping_segments(
        self,
        segments: List[Dict[str, Any]],
        overlap_threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Remove overlapping segments that are too close in time.
        
        Args:
            segments: List of segments
            overlap_threshold: Time threshold in seconds
            
        Returns:
            List of non-overlapping segments
        """
        if not segments:
            return []
            
        filtered_segments = [segments[0]]
        
        for segment in segments[1:]:
            if segment['start_time'] - filtered_segments[-1]['start_time'] > overlap_threshold:
                filtered_segments.append(segment)
        
        return filtered_segments
    
    def process_video(
        self,
        video_url: str,
        query: str
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Process a YouTube video and find relevant segments.
        
        Args:
            video_url: YouTube video URL
            query: Search query
            
        Returns:
            Tuple of (relevant segments, error message if any)
        """
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return [], "Invalid YouTube URL"
            
        transcript = self.get_transcript(video_id)
        if not transcript:
            return [], "Could not fetch video transcript"
            
        relevant_segments = self.find_relevant_segments(transcript, query)
        if not relevant_segments:
            return [], "No relevant segments found"
            
        return relevant_segments, None 