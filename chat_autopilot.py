#!/usr/bin/env python3
"""
Interactive chat to analyze autopilot decisions.
Load a log file and ask LLM questions about specific moments.
"""

import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from openai import AzureOpenAI

from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)


class AutopilotChat:
    """Interactive chat for analyzing autopilot decisions."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.entries: List[Dict] = []
        self.current_entry: Optional[Dict] = None
        self.conversation_history: List[Dict] = []
        
        # Initialize Azure OpenAI
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        
        self._load_log()
        
    def _load_log(self):
        """Load JSONL log file."""
        if not os.path.exists(self.log_file):
            print(f"❌ Log file not found: {self.log_file}")
            return
            
        with open(self.log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        self.entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        print(f"✅ Loaded {len(self.entries)} entries from {self.log_file}")
        
    def list_entries(self, start: int = 0, count: int = 20):
        """List log entries with timestamps."""
        print(f"\n{'='*70}")
        print(f"{'#':<6} {'Timestamp':<12} {'Frame':<8} {'Objects':<8} {'Pedal':<8} {'Gear':<6} {'Reasoning'}")
        print(f"{'='*70}")
        
        for i, entry in enumerate(self.entries[start:start+count], start=start):
            timestamp = entry.get('timestamp', 'N/A')
            frame = entry.get('frame', 'N/A')
            objects = len(entry.get('detected_objects', []))
            cmd = entry.get('command', {})
            pedal = cmd.get('pedal', '?')
            pedal_pct = cmd.get('pedal_percent', 0)
            gear = cmd.get('gear', '?')
            reasoning = cmd.get('reasoning', '')[:30]
            
            # Format timestamp nicely
            if isinstance(timestamp, str) and 'T' in timestamp:
                ts_short = timestamp.split('T')[1][:8]
            else:
                ts_short = str(timestamp)[:12]
            
            print(f"{i:<6} {ts_short:<12} {frame:<8} {objects:<8} {pedal}:{pedal_pct}%{'':<3} {gear:<6} {reasoning}...")
            
        print(f"\nShowing {start}-{min(start+count, len(self.entries))} of {len(self.entries)} entries")
        print("Use 'goto <number>' to select an entry, 'next'/'prev' to browse")
        
    def select_entry(self, index: int):
        """Select an entry for discussion."""
        if 0 <= index < len(self.entries):
            self.current_entry = self.entries[index]
            self.conversation_history = []  # Reset conversation
            self._show_entry_details()
            return True
        else:
            print(f"❌ Invalid index. Valid range: 0-{len(self.entries)-1}")
            return False
            
    def _show_entry_details(self):
        """Show details of current entry."""
        if not self.current_entry:
            print("❌ No entry selected")
            return
            
        entry = self.current_entry
        print(f"\n{'='*70}")
        print(f"📍 SELECTED ENTRY")
        print(f"{'='*70}")
        print(f"Timestamp: {entry.get('timestamp', 'N/A')}")
        print(f"Frame: {entry.get('frame', 'N/A')}")
        
        print(f"\n🎯 Detected Objects:")
        objects = entry.get('detected_objects', [])
        if objects:
            for obj in objects:
                if isinstance(obj, dict):
                    name = obj.get('class_name', obj.get('name', '?'))
                    pos = obj.get('position', '?')
                    dist = obj.get('distance_estimate', '?')
                    dist_m = obj.get('distance_meters', obj.get('distance_m', '?'))
                    conf = obj.get('confidence', 0)
                    print(f"  - {name}: position={pos}, distance={dist}, {dist_m}m, confidence={conf:.0%}")
                else:
                    print(f"  - {obj}")
        else:
            print("  (no objects detected)")
            
        cmd = entry.get('command', {})
        print(f"\n🎮 Command:")
        print(f"  Steering: {cmd.get('steering', '?')} ({cmd.get('steering_angle', 0)}°)")
        print(f"  Pedal: {cmd.get('pedal', '?')} @ {cmd.get('pedal_percent', 0)}%")
        print(f"  Gear: {cmd.get('gear', '?')}")
        print(f"  Reasoning: {cmd.get('reasoning', 'N/A')}")
        
        print(f"\n💬 You can now ask questions about this decision.")
        print(f"   Example: 'Why did you choose brake instead of gas?'")
        print(f"   Example: 'What if the pedestrian was further away?'")
        print(f"{'='*70}")
        
    def _build_context(self) -> str:
        """Build context string from current entry."""
        if not self.current_entry:
            return "No entry selected."
            
        entry = self.current_entry
        
        # Build objects description
        objects = entry.get('detected_objects', [])
        if objects:
            obj_lines = []
            for obj in objects:
                if isinstance(obj, dict):
                    name = obj.get('class_name', obj.get('name', '?'))
                    pos = obj.get('position', '?')
                    dist = obj.get('distance_estimate', '?')
                    dist_m = obj.get('distance_meters', obj.get('distance_m', '?'))
                    conf = obj.get('confidence', 0)
                    obj_lines.append(f"- {name}: position={pos}, distance={dist}, {dist_m}m, confidence={conf:.0%}")
                else:
                    obj_lines.append(f"- {obj}")
            objects_text = "\n".join(obj_lines)
        else:
            objects_text = "No objects detected"
            
        cmd = entry.get('command', {})
        
        context = f"""
SITUATION AT FRAME {entry.get('frame', 'N/A')} (timestamp: {entry.get('timestamp', 'N/A')}):

Detected objects:
{objects_text}

Decision made by autopilot:
- Steering: {cmd.get('steering', '?')} at {cmd.get('steering_angle', 0)} degrees
- Pedal: {cmd.get('pedal', '?')} at {cmd.get('pedal_percent', 0)}%
- Gear: {cmd.get('gear', '?')}
- Reasoning: {cmd.get('reasoning', 'N/A')}
"""
        return context
        
    def ask(self, question: str) -> str:
        """Ask a question about the current entry."""
        if not self.current_entry:
            return "❌ Please select an entry first using 'goto <number>'"
            
        context = self._build_context()
        
        system_prompt = """You are an AI assistant explaining autopilot decisions for a toy car.
You have access to the situation data (detected objects, positions, distances) and the decision that was made.

When answering questions:
1. Explain WHY the decision was made based on the detected objects
2. Reference specific objects, their positions and distances
3. Explain what alternative decisions could have been made
4. Be honest if a decision seems suboptimal

Keep answers concise but informative. Speak in first person as if you made the decision."""

        # Add context as first user message if conversation is new
        if not self.conversation_history:
            self.conversation_history.append({
                "role": "user",
                "content": f"Here is the situation I want to discuss:\n{context}"
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": "I understand. I can see the situation and the decision I made. What would you like to know about this moment?"
            })
        
        # Add user question
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.conversation_history
                ],
                temperature=0.7,
                max_tokens=500,
            )
            
            answer = response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })
            
            return answer
            
        except Exception as e:
            return f"❌ Error: {e}"
            
    def find_by_timestamp(self, time_str: str) -> List[int]:
        """Find entries matching a timestamp pattern."""
        matches = []
        for i, entry in enumerate(self.entries):
            ts = str(entry.get('timestamp', ''))
            if time_str in ts:
                matches.append(i)
        return matches
        
    def find_by_action(self, action: str) -> List[int]:
        """Find entries where specific action was taken."""
        matches = []
        action = action.lower()
        for i, entry in enumerate(self.entries):
            cmd = entry.get('command', {})
            pedal = cmd.get('pedal', '').lower()
            reasoning = cmd.get('reasoning', '').lower()
            
            if action in pedal or action in reasoning:
                matches.append(i)
        return matches
        
    def run(self):
        """Run interactive chat loop."""
        print("\n" + "="*70)
        print("🤖 AUTOPILOT DECISION ANALYZER")
        print("="*70)
        print("\nCommands:")
        print("  list [start]     - List entries (default: first 20)")
        print("  goto <number>    - Select entry by number")
        print("  next / prev      - Browse entries")
        print("  find <text>      - Find entries by timestamp or action")
        print("  show             - Show current entry details")
        print("  ask <question>   - Ask about current entry (or just type question)")
        print("  quit             - Exit")
        print("\n" + "="*70)
        
        if self.entries:
            self.list_entries(0, 10)
        
        current_page = 0
        page_size = 20
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            # Parse commands
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            elif cmd == 'list':
                start = int(arg) if arg.isdigit() else current_page * page_size
                self.list_entries(start, page_size)
                
            elif cmd == 'next':
                current_page += 1
                start = current_page * page_size
                if start >= len(self.entries):
                    current_page = 0
                    start = 0
                self.list_entries(start, page_size)
                
            elif cmd == 'prev':
                current_page = max(0, current_page - 1)
                self.list_entries(current_page * page_size, page_size)
                
            elif cmd == 'goto':
                if arg.isdigit():
                    self.select_entry(int(arg))
                else:
                    print("Usage: goto <number>")
                    
            elif cmd == 'show':
                self._show_entry_details()
                
            elif cmd == 'find':
                if not arg:
                    print("Usage: find <timestamp or action>")
                    continue
                    
                # Try timestamp first
                matches = self.find_by_timestamp(arg)
                if not matches:
                    matches = self.find_by_action(arg)
                    
                if matches:
                    print(f"Found {len(matches)} matches: {matches[:20]}...")
                    if len(matches) == 1:
                        self.select_entry(matches[0])
                else:
                    print(f"No matches found for '{arg}'")
                    
            elif cmd == 'ask':
                if arg:
                    answer = self.ask(arg)
                    print(f"\n🤖 Assistant: {answer}")
                else:
                    print("Usage: ask <question>")
                    
            else:
                # Treat as question if entry is selected
                if self.current_entry:
                    answer = self.ask(user_input)
                    print(f"\n🤖 Assistant: {answer}")
                else:
                    print("❓ Unknown command. Select an entry first with 'goto <number>', then ask questions.")


def main():
    parser = argparse.ArgumentParser(description="Chat with autopilot about its decisions")
    parser.add_argument(
        '--log', '-l',
        type=str,
        help='Path to JSONL log file'
    )
    parser.add_argument(
        '--list-logs',
        action='store_true',
        help='List available log files'
    )
    
    args = parser.parse_args()
    
    log_dir = "autopilot_logs"
    
    if args.list_logs or not args.log:
        # List available logs
        if os.path.exists(log_dir):
            logs = sorted([f for f in os.listdir(log_dir) if f.endswith('.jsonl')])
            if logs:
                print("\n📁 Available log files:")
                for i, log in enumerate(logs):
                    path = os.path.join(log_dir, log)
                    size = os.path.getsize(path)
                    with open(path) as f:
                        lines = sum(1 for _ in f)
                    print(f"  {i+1}. {log} ({lines} entries, {size//1024}KB)")
                    
                if not args.log:
                    print("\nUsage: python chat_autopilot.py --log autopilot_logs/<filename>.jsonl")
                    return
            else:
                print("No log files found in autopilot_logs/")
                return
        else:
            print("No autopilot_logs directory found. Run autopilot first.")
            return
            
    if args.log:
        chat = AutopilotChat(args.log)
        chat.run()


if __name__ == "__main__":
    main()

