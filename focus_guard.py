"""
FocusGuard: Privacy-Focused AI Distraction Blocker
-------------------------------------------------
An AI-powered application that monitors screen activity, classifies productivity,
and helps maintain focus through smart interventions while respecting privacy.
"""

import os
import time
import json
import logging
import platform
import subprocess
import re
import functools
import threading
import queue
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import pyautogui
import pytesseract
from PIL import Image
import numpy as np
import cv2
import requests
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import textwrap
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("focus_guard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FocusGuard")

# Configuration
DEFAULT_CONFIG = {
    "screenshot_interval": 8,  # seconds
    "ocr_psm_mode": 6,
    "max_text_length": 500,  # token limit for API
    "distraction_threshold": 3,  # consecutive distractions before intervention
    "api_timeout": 10,  # seconds
    "notification_frequency": 2,  # minutes
    "enable_voice_alerts": True,
    "enable_app_blocking": True,
    "privacy": {
        "store_screenshots": False,
        "store_raw_text": False,
        "anonymize_personal_data": True
    },
    "distractions": {
        "social_media": ["instagram.com", "facebook.com", "twitter.com", "tiktok.com", "reddit.com"],
        "entertainment": ["netflix.com", "youtube.com/watch", "twitch.tv"],
        "games": ["steam", "epic games", "battle.net"]
    },
    "productivity": {
        "work": ["vscode", "terminal", "cmd.exe", "powershell", "excel", "word", "powerpoint", "google docs"],
        "study": ["pdf", "research", "lecture", "coursera", "edx", "khan academy"],
        "communication": ["outlook", "gmail", "calendar", "meets", "zoom"]
    }
}

# ===== SCREEN MONITORING MODULE =====

class ScreenMonitor:
    """Handles screenshot capture and text extraction."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_capture_time = 0
        self.screenshot_interval = config.get("screenshot_interval", 8)
        self.ocr_psm_mode = config.get("ocr_psm_mode", 6)
        
        # Ensure tesseract is properly configured
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def capture_screen(self) -> Image.Image:
        """Capture the current screen."""
        screenshot = pyautogui.screenshot()
        return screenshot
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get black and white image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        return thresh
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from the preprocessed image."""
        custom_config = f'--psm {self.ocr_psm_mode}'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text
    
    def get_active_window_text(self) -> str:
        """Get text from active window with rate limiting."""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_capture_time < self.screenshot_interval:
            time_to_wait = self.screenshot_interval - (current_time - self.last_capture_time)
            time.sleep(max(0, time_to_wait))
        
        # Capture and process
        screenshot = self.capture_screen()
        preprocessed = self.preprocess_image(screenshot)
        text = self.extract_text(preprocessed)
        
        # Clean and limit text
        cleaned_text = self.clean_text(text)
        
        # Update timestamp
        self.last_capture_time = time.time()
        
        return cleaned_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Filter out common non-informative elements
        text = re.sub(r'[^\w\s.,?!:\-]', '', text)
        
        # Limit text length
        if len(text) > self.config.get("max_text_length", 500):
            text = text[:self.config.get("max_text_length", 500)]
        
        return text

    def get_active_window_info(self) -> Dict[str, str]:
        """Get both text content and window title"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_capture_time < self.screenshot_interval:
            time_to_wait = self.screenshot_interval - (current_time - self.last_capture_time)
            time.sleep(max(0, time_to_wait))
        
        # Get window title (platform specific)
        window_title = ""
        try:
            if platform.system() == "Windows":
                import win32gui
                window_title = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            elif platform.system() == "Darwin":
                window_title = subprocess.run(
                    ["osascript", "-e", 'tell app "System Events" to get name of first process whose frontmost is true'],
                    capture_output=True, text=True
                ).stdout.strip()
            else:  # Linux
                window_title = subprocess.run(
                    ["xdotool", "getwindowfocus", "getwindowname"],
                    capture_output=True, text=True
                ).stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get window title: {e}")
        
        # Get screen content
        screenshot = self.capture_screen()
        preprocessed = self.preprocess_image(screenshot)
        text = self.extract_text(preprocessed)
        cleaned_text = self.clean_text(text)
        
        self.last_capture_time = time.time()
        
        return {
            "text": cleaned_text,
            "window_title": window_title,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# ===== AI CLASSIFICATION MODULE =====

class ActivityClassifier:
    """Classifies screen activity using Meta-Llama-3-2-3B-Instruct via Akash API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv("AKASH_API_KEY")
        if not self.api_key:
            logger.warning("AKASH_API_KEY not found in environment variables!")
        
        self.api_base_url = "https://chatapi.akash.network/api/v1"
        self.timeout = config.get("api_timeout", 10)
        self.prompt_template = textwrap.dedent("""
            Classify this screen activity as: [STUDY] | [SOCIAL_MEDIA] | [NEUTRAL] | [WORK] | [ENTERTAINMENT] | [GAMES]
            
            Rules:
            - STUDY: Educational content, research papers, learning platforms, textbooks, PDFs with academic content
            - WORK: Productivity tools (VS Code, Terminal), business applications, work documents
            - SOCIAL_MEDIA: Instagram, TikTok, Reddit, Twitter, Facebook, social chats
            - ENTERTAINMENT: YouTube videos (non-educational), Netflix, streaming
            - GAMES: Any gaming content, launchers (Steam, Epic Games, Battle.net), game stores, or actual games
            - NEUTRAL: Email, Calendar, Maps, system applications
            
            Content: "{text}"
            
            Return ONLY the classification label without explanation.
        """)
        self.classify_with_cache = functools.lru_cache(maxsize=100)(self._classify_activity)
    
    def _classify_activity(self, text: str) -> str:
        """Send text to Meta-Llama-3 via Akash API."""
        if not text.strip():
            return "NEUTRAL"
        
        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url
            )
            
            response = client.chat.completions.create(
                model="Meta-Llama-3-2-3B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt_template.format(text=text)
                    }
                ],
                temperature=0.3,
                max_tokens=10,
            )
            
            label = response.choices[0].message.content.strip()
            # Extract label if wrapped in brackets (e.g., "[WORK]")
            if "[" in label and "]" in label:
                label = re.search(r'\[(.*?)\]', label).group(1)
            return label
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return "ERROR"
    
    def classify_activity(self, window_data: Dict[str, str]) -> str:
        """Classify using both window title and content"""
        text = window_data.get("text", "")
        window_title = window_data.get("window_title", "").lower()
        text_lower = text.lower()
        
        # Combined analysis
        combined_text = f"{window_title} {text_lower}"
        
        # Game detection (enhanced)
        game_indicators = [
            "epic games", "steam", "battle.net", "origin", "uplay",
            "fortnite", "minecraft", "league of legends", "game", "launcher"
        ]
        
        if any(indicator in combined_text for indicator in game_indicators):
            return "GAMES"
        
        # Browser detection
        browser_indicators = {
            "chrome": ["google chrome", "chrome.exe"],
            "firefox": ["mozilla firefox", "firefox.exe"],
            "edge": ["microsoft edge", "msedge.exe"]
        }
        
        current_browser = None
        for browser, indicators in browser_indicators.items():
            if any(indicator in combined_text for indicator in indicators):
                current_browser = browser
                break
                
        # Website detection (if in browser)
        if current_browser:
            website = self._extract_website(window_title)
            if website:
                return self._classify_website(website)
        
        # Check for known patterns in text
        text_lower = text.lower()
        
        # Match against distraction patterns
        for category, patterns in self.config.get("distractions", {}).items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    if category == "social_media":
                        return "SOCIAL_MEDIA"
                    elif category == "entertainment":
                        return "ENTERTAINMENT"
        
        # Match against productivity patterns
        for category, patterns in self.config.get("productivity", {}).items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    if category == "work":
                        return "WORK"
                    elif category == "study":
                        return "STUDY"
                    elif category == "communication":
                        return "NEUTRAL"
        
        # Fall back to AI classification
        return self.classify_with_cache(text)
    
    def _extract_website(self, window_title: str) -> Optional[str]:
        """Extract website from browser window title"""
        patterns = [
            r"(https?://[^\s/$.?#].[^\s]*)",  # Full URLs
            r"([\w-]+\.(com|org|net|io))"     # Domain names
        ]
        
        for pattern in patterns:
            match = re.search(pattern, window_title)
            if match:
                return match.group(1)
        return None
    
    def _classify_website(self, url: str) -> str:
        """Classify specific websites"""
        url = url.lower()
        
        for category, patterns in self.config.get("distractions", {}).items():
            for pattern in patterns:
                if pattern.lower() in url:
                    if category == "social_media":
                        return "SOCIAL_MEDIA"
                    elif category == "entertainment":
                        return "ENTERTAINMENT"
        
        return "NEUTRAL"

# ===== INTERVENTION MODULE =====

class InterventionManager:
    """Manages intervention actions when distractions are detected."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.distraction_count = 0
        self.last_notification_time = 0
        self.notification_frequency = config.get("notification_frequency", 2) * 60  # convert to seconds
        self.enable_voice_alerts = config.get("enable_voice_alerts", True)
        self.enable_app_blocking = config.get("enable_app_blocking", True)
        self.temp_dir = "temp"
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def handle_activity(self, activity_type: str, window_data: Dict[str, str]) -> Dict[str, Any]:
        """Process the detected activity and take appropriate action."""
        timestamp = window_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        result = {
            "timestamp": timestamp,
            "activity_type": activity_type,
            "content_preview": window_data.get("text", "")[:100] + "..." if len(window_data.get("text", "")) > 100 else window_data.get("text", ""),
            "window_title": window_data.get("window_title", ""),
            "action_taken": None
        }
        
        # Reset counter for productive activities
        if activity_type in ["STUDY", "WORK", "NEUTRAL"]:
            self.distraction_count = 0
            return result
        
        # Increment counter for distractions
        self.distraction_count += 1
        result["distraction_count"] = self.distraction_count
        
        # Take action if threshold reached
        if self.distraction_count >= self.config.get("distraction_threshold", 3):
            current_time = time.time()
            
            # Check if we should notify based on frequency setting
            if current_time - self.last_notification_time >= self.notification_frequency:
                # Send notification
                self.send_notification(activity_type, window_data.get("window_title", ""))
                result["action_taken"] = "notification"
                
                # Voice alert
                if self.enable_voice_alerts and activity_type:
                    self.play_voice_alert(activity_type)
                    result["action_taken"] = "voice_alert"
                
                # Tab/App blocking
                if self.enable_app_blocking:
                    # Special handling for games
                    if activity_type == "GAMES":
                        app_blocked = self.block_game(window_data)
                    else:
                        app_blocked = self.block_distraction(activity_type, window_data)
                        
                    if app_blocked:
                        result["action_taken"] = "tab_closed"
                        result["app_blocked"] = app_blocked
                
                # Reset counter and update notification time
                self.distraction_count = 0
                self.last_notification_time = current_time
        
        return result
    
    def block_game(self, window_data: Dict[str, str]) -> Optional[str]:
        """Close game launchers and applications."""
        if not self.enable_app_blocking:
            return None
            
        content = window_data.get("window_title", "") + " " + window_data.get("text", "")
        content_lower = content.lower()
        game_blocked = None
        
        # Map of game launchers to detect and block
        game_launchers = {
            'steam': 'Steam',
            'epic': 'Epic Games',
            'epicgames': 'Epic Games',
            'battle.net': 'Battle.net',
            'battlenet': 'Battle.net',
            'origin': 'Origin',
            'uplay': 'Uplay',
            'ubisoft': 'Ubisoft Connect',
            'gog': 'GOG Galaxy',
            'xbox': 'Xbox App'
        }

        # Check for matches
        for keyword, description in game_launchers.items():
            if keyword in content_lower:
                game_blocked = description
                self._terminate_application(game_blocked)
                return game_blocked
                
        # If no specific launcher found but still classified as game
        if 'game' in content_lower:
            self._terminate_application("game")
            return "Game Application"
            
        return None
    
    def send_notification(self, activity_type: str, content: str) -> None:
        """Send system notification about distraction."""
        if not activity_type:
            return
            
        try:
            message = f"Detected {activity_type.lower()} distraction"
            if content:
                message += f": {content[:50]}{'...' if len(content) > 50 else ''}"
            
            if platform.system() == "Windows":
                try:
                    from plyer import notification
                    notification.notify(
                        title="⚠️ Focus Guard Alert",
                        message=message,
                        timeout=5
                    )
                    return
                except ImportError:
                    subprocess.run(["msg", "*", message], shell=True)
            elif platform.system() == "Darwin":
                subprocess.run(["osascript", "-e", f'display notification "{message}" with title "FocusGuard Alert"'])
            else:  # Linux
                subprocess.run(["notify-send", "FocusGuard Alert", message])
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def play_voice_alert(self, activity_type: str) -> None:
        """Play voice alert about distraction."""
        if not activity_type:
            return
            
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                mp3_path = temp_mp3.name
                
            try:
                # Generate TTS and save as MP3
                tts = gTTS(
                    text=f"Warning! {activity_type} distraction detected",
                    lang='en',
                    slow=False
                )
                tts.save(mp3_path)
                
                # Play audio in a separate thread
                def _play_audio():
                    try:
                        sound = AudioSegment.from_mp3(mp3_path)
                        play(sound)
                    except Exception as e:
                        logger.error(f"Audio playback failed: {e}")
                        if platform.system() == "Windows":
                            import winsound
                            winsound.Beep(1000, 500)  # Fallback beep
                    finally:
                        # Cleanup file
                        if os.path.exists(mp3_path):
                            try:
                                os.remove(mp3_path)
                            except Exception:
                                pass

                # Run playback in a separate thread
                threading.Thread(target=_play_audio, daemon=True).start()

            except Exception as e:
                # Cleanup on error
                if os.path.exists(mp3_path):
                    try:
                        os.remove(mp3_path)
                    except Exception:
                        pass
                raise e

        except Exception as e:
            logger.error(f"Voice alert system failed: {e}")
            self.send_notification(activity_type, "Distraction detected (voice failed)")
            if platform.system() == "Windows":
                try:
                    import winsound
                    winsound.Beep(1000, 500)
                except Exception:
                    pass
        
    def block_distraction(self, activity_type: str, window_data: Dict[str, str]) -> Optional[str]:
        """Close specific tabs based on detected content."""
        if not self.enable_app_blocking:
            return None
            
        content = window_data.get("window_title", "") + " " + window_data.get("text", "")
        content_lower = content.lower()
        site_blocked = None
        
        # Map of distraction sites to identify specific tabs to close
        distraction_sites = {
            'instagram.com': 'Instagram',
            'facebook.com': 'Facebook',
            'twitter.com': 'Twitter',
            'tiktok.com': 'TikTok',
            'reddit.com': 'Reddit',
            'youtube.com/watch': 'YouTube',
            'netflix.com': 'Netflix',
            'twitch.tv': 'Twitch'
        }
        
        # Check for matches
        for site, description in distraction_sites.items():
            if site in content_lower:
                site_blocked = description
                self._close_specific_tab(site)
                return site_blocked
                
        # Check for gaming platforms
        game_platforms = ['steam', 'epic games','epicgames','launcher', 'battle.net']
        for platform in game_platforms:
            if platform in content_lower:
                # For game launchers, we still close the application
                self._terminate_application(platform)
                return platform
            
        return None
    
    def _close_specific_tab(self, target_site: str) -> None:
        """Close a specific browser tab containing the target site."""
        try:
            # Different approaches based on operating system
            if platform.system() == "Windows":
                # Method 1: Use browser keyboard shortcuts
                # First bring the window to focus
                pyautogui.hotkey('alt', 'tab')
                time.sleep(0.5)
                
                # Then close the current tab
                pyautogui.hotkey('ctrl', 'w')
                logger.info(f"Attempted to close tab containing {target_site} using keyboard shortcut")
                
            elif platform.system() == "Darwin":  # macOS
                # For macOS, use Command+W to close tab
                pyautogui.hotkey('command', 'w')
                logger.info(f"Attempted to close tab containing {target_site} using keyboard shortcut")
                
            else:  # Linux
                # For Linux, use Ctrl+W to close tab
                pyautogui.hotkey('ctrl', 'w')
                logger.info(f"Attempted to close tab containing {target_site} using keyboard shortcut")
            
        except Exception as e:
            logger.error(f"Failed to close tab for {target_site}: {str(e)}")
    
    def _terminate_application(self, app_type: str) -> None:
        """Terminate the specified application type."""
        try:
            if platform.system() == "Windows":
                if app_type.lower() in ["steam", "epic games", "epicgames", "battle.net"]:
                    # Terminate game launchers
                    for launcher in ["steam.exe", "epicgameslauncher.exe", "battle.net.exe"]:
                        subprocess.run(["taskkill", "/F", "/IM", launcher],
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
            
            elif platform.system() == "Darwin":  # macOS
                if app_type.lower() in ["steam", "epic games", "epicgames", "battle.net"]:
                    # Terminate game launchers
                    for launcher in ["Steam", "Epic Games Launcher", "Battle.net"]:
                        subprocess.run(["pkill", "-f", launcher],
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
            
            else:  # Linux
                if app_type.lower() in ["steam", "epic games", "epicgames", "battle.net"]:
                    # Terminate game launchers
                    subprocess.run(["pkill", "-f", "steam"],
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    subprocess.run(["pkill", "-f", "epicgames"],
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    
        except Exception as e:
            logger.error(f"Failed to terminate {app_type}: {str(e)}")
    
    def update_settings(self, config: Dict[str, Any]) -> None:
        """Update intervention settings from config."""
        self.config = config
        self.notification_frequency = config.get("notification_frequency", 2) * 60  # convert to seconds
        self.enable_voice_alerts = config.get("enable_voice_alerts", True)
        self.enable_app_blocking = config.get("enable_app_blocking", True)

# ===== DATA MANAGEMENT MODULE =====

class ActivityTracker:
    """Tracks and stores activity history with privacy controls."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.activities = []
        self.max_history = 1000  # Maximum activities to keep in memory
        self.privacy_settings = config.get("privacy", {})
        self.data_file = "focus_data.json"
    
    def add_activity(self, activity_data: Dict[str, Any]) -> None:
        """Add an activity entry to tracking history."""
        # Apply privacy controls
        if self.privacy_settings.get("anonymize_personal_data", True):
            activity_data = self._anonymize_data(activity_data)
            
        # Store only what's allowed by privacy settings
        if not self.privacy_settings.get("store_raw_text", False):
            if "content_preview" in activity_data:
                # Keep only first few words as a preview
                words = activity_data["content_preview"].split()
                activity_data["content_preview"] = " ".join(words[:5]) + "..." if len(words) > 5 else activity_data["content_preview"]
        
        # Add to history
        self.activities.append(activity_data)
        
        # Trim history if needed
        if len(self.activities) > self.max_history:
            self.activities = self.activities[-self.max_history:]
    
    def _anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize potentially sensitive information."""
        anonymized = data.copy()
        
        if "content_preview" in anonymized:
            # Replace emails
            anonymized["content_preview"] = re.sub(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "[EMAIL]",
                anonymized["content_preview"]
            )
            
            # Replace phone numbers
            anonymized["content_preview"] = re.sub(
                r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
                "[PHONE]",
                anonymized["content_preview"]
            )
            
            # Replace URLs
            anonymized["content_preview"] = re.sub(
                r'https?://\S+',
                "[URL]",
                anonymized["content_preview"]
            )
        
        return anonymized
    
    def get_recent_activities(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent activity history."""
        return self.activities[-count:] if self.activities else []
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics from activity history."""
        if not self.activities:
            return {
                "total_tracked_time": 0,
                "activity_breakdown": {},
                "intervention_count": 0,
                "productivity_score": 0
            }
        
        # Count activity types
        activity_counts = {}
        intervention_count = 0
        
        for activity in self.activities:
            activity_type = activity.get("activity_type", "UNKNOWN")
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
            
            if activity.get("action_taken"):
                intervention_count += 1
        
        # Calculate total tracked time (rough estimate)
        total_time_minutes = len(self.activities) * (self.config.get("screenshot_interval", 8) / 60)
        
        # Calculate productivity score (0-100)
        productive_types = ["STUDY", "WORK", "NEUTRAL"]
        productive_count = sum(activity_counts.get(t, 0) for t in productive_types)
        productivity_score = int((productive_count / len(self.activities)) * 100) if self.activities else 0
        
        return {
            "total_tracked_time": total_time_minutes,
            "activity_breakdown": activity_counts,
            "intervention_count": intervention_count,
            "productivity_score": productivity_score
        }
    
    def save_data(self) -> None:
        """Save activity data to file if permitted by privacy settings."""
        if not self.privacy_settings.get("store_raw_text", False):
            # Save only summary data
            summary = self.get_summary_stats()
            try:
                with open(self.data_file, 'w') as f:
                    json.dump(summary, f)
            except Exception as e:
                logger.error(f"Failed to save data: {str(e)}")
        else:
            # Save full activity data
            try:
                with open(self.data_file, 'w') as f:
                    json.dump(self.activities, f)
            except Exception as e:
                logger.error(f"Failed to save data: {str(e)}")
    
    def load_data(self) -> None:
        """Load saved activity data if available."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.activities = data
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")

# ===== MAIN APPLICATION =====

class FocusGuard:
    """Main application class that coordinates all modules."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.screen_monitor = ScreenMonitor(self.config)
        self.activity_classifier = ActivityClassifier(self.config)
        self.intervention_manager = InterventionManager(self.config)
        self.activity_tracker = ActivityTracker(self.config)
        
        self.running = False
        self.monitoring_thread = None
        self.event_queue = queue.Queue()
    
    def start_monitoring(self) -> None:
        """Start the monitoring process in a separate thread."""
        if self.running:
            logger.warning("Monitoring already running")
            return
            
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Focus monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring process."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        
        # Save activity data before exiting
        self.activity_tracker.save_data()
        logger.info("Focus monitoring stopped")
    
    def _monitoring_loop(self):
        while self.running:
            try:
                # Get window info
                window_data = self.screen_monitor.get_active_window_info()
                
                # Classify activity with the full window data
                activity_type = self.activity_classifier.classify_activity(window_data)
                
                # Handle the detected activity with full context
                result = self.intervention_manager.handle_activity(activity_type, window_data)
                
                # Track the activity
                self.activity_tracker.add_activity(result)
                
                # Add to event queue
                self.event_queue.put(result)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get new events from the queue (non-blocking)."""
        events = []
        while not self.event_queue.empty():
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.activity_tracker.get_summary_stats()
    
    def get_recent_activities(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent activity history."""
        return self.activity_tracker.get_recent_activities(count)
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration settings."""
        self.config.update(new_config)
        
        # Update component configs
        self.screen_monitor.config.update(new_config)
        self.activity_classifier.config.update(new_config)
        
        # Update intervention manager
        self.intervention_manager.config.update(new_config)
        self.intervention_manager.enable_app_blocking = new_config.get("enable_app_blocking", True)
        self.intervention_manager.enable_voice_alerts = new_config.get("enable_voice_alerts", True)
        self.intervention_manager.notification_frequency = new_config.get("notification_frequency", 2) * 60  # convert to seconds
        
        # Update activity tracker settings
        self.activity_tracker.privacy_settings = new_config.get("privacy", {})
        
        logger.info("Configuration updated")