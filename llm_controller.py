"""
LLM integration module for generating control commands.

Supports multiple providers: Azure OpenAI, Ollama (local), OpenAI.
Takes object detection results and generates autopilot commands.
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from openai import AzureOpenAI, OpenAI

from config import (
    LLM_PROVIDER,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    CLASSES,
)


@dataclass
class DetectedObject:
    """Detected object in the image."""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixels
    position: str  # "left", "center", "right"
    distance_estimate: str  # "close", "medium", "far"
    distance_meters: float = 0.0  # estimated distance in meters


@dataclass
class ControlCommand:
    """Control command for autopilot."""
    steering: str  # "left", "right", "center"
    steering_angle: int  # 0-90 degrees
    pedal: str  # "brake", "gas"
    pedal_percent: int  # 0-100%
    gear: int  # 1-5
    reasoning: str  # explanation of decision


class LLMController:
    """
    Controller for generating control commands via LLM.
    """
    
    def __init__(self, debug: bool = False, provider: str = None):
        """Initialize LLM client.
        
        Args:
            debug: If True, print LLM prompts to console
            provider: Override LLM_PROVIDER from config ("azure", "ollama", "openai")
        """
        self.debug = debug
        self.provider = provider or LLM_PROVIDER
        
        if self.provider == "azure":
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
            )
            self.model = AZURE_OPENAI_DEPLOYMENT
            print(f"🔵 LLM: Azure OpenAI ({self.model})")
            
        elif self.provider == "ollama":
            # Ollama uses OpenAI-compatible API
            self.client = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama",  # Ollama doesn't need real API key
            )
            self.model = OLLAMA_MODEL
            print(f"🟢 LLM: Ollama local ({self.model})")
            
        elif self.provider == "openai":
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
            )
            self.model = OPENAI_MODEL
            print(f"🟡 LLM: OpenAI ({self.model})")
            
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
        
        # Keep for backward compatibility
        self.deployment = self.model
        
        # Use simplified prompt for small local models
        use_simple_prompt = self.provider == "ollama" and any(
            x in self.model.lower() for x in ["3b", "1b", "2b", "mini", "tiny", "small"]
        )
        
        if use_simple_prompt:
            print("📝 Using simplified prompt for small model")
            self.system_prompt = self._get_simple_prompt()
        else:
            self.system_prompt = self._get_full_prompt()
    
    def _get_simple_prompt(self) -> str:
        """Simplified prompt for small models (3B and less) with few-shot examples."""
        return """You control a car. Read the scene, output JSON with steering!

ROAD FOLLOWING (CRITICAL!):
- movement_angle shows CURRENT road direction
- |movement_angle| < 5° → STRAIGHT ROAD → steering=CENTER, angle=0 !!!
- movement_angle < -10° → road curves LEFT → steer LEFT
- movement_angle > 10° → road curves RIGHT → steer RIGHT

STRAIGHT ROAD RULE (VERY IMPORTANT!):
- When |movement_angle| < 5° → ALWAYS steer CENTER with angle 0!
- Don't keep turning when road is straight!

EXAMPLES:

Scene: movement_angle=2, no objects
{"steering":"center","steering_angle":0,"pedal":"gas","pedal_percent":45,"gear":2,"reasoning":"straight road, center"}

Scene: movement_angle=-3, car at center 12m
{"steering":"center","steering_angle":0,"pedal":"gas","pedal_percent":35,"gear":2,"reasoning":"straight road, following car"}

Scene: movement_angle=1, car at left 5m
{"steering":"center","steering_angle":0,"pedal":"gas","pedal_percent":40,"gear":2,"reasoning":"straight road, car on left side"}

Scene: movement_angle=-15, no objects
{"steering":"left","steering_angle":20,"pedal":"gas","pedal_percent":40,"gear":2,"reasoning":"road curves left, following"}

Scene: movement_angle=25, car at right 8m
{"steering":"right","steering_angle":30,"pedal":"gas","pedal_percent":35,"gear":2,"reasoning":"road curves right, following curve"}

Scene: movement_angle=-8, car at center 4m, left clear
{"steering":"left","steering_angle":20,"pedal":"gas","pedal_percent":25,"gear":2,"reasoning":"avoiding car, slight left curve"}

Scene: movement_angle=35, no objects
{"steering":"right","steering_angle":40,"pedal":"gas","pedal_percent":30,"gear":2,"reasoning":"sharp right curve"}

RULES:
1. |movement_angle| < 5° = STRAIGHT = steering CENTER, angle 0!
2. |movement_angle| 5-15° = gentle curve, steer 10-20°
3. |movement_angle| > 15° = curve, steer 20-40°
4. |movement_angle| > 30° = sharp, steer 35-50°
5. Object CENTER < 5m = steer away OR brake

Output ONLY JSON!"""

    def _get_full_prompt(self) -> str:
        """Full prompt for large models (7B+)."""
        return """You are an autopilot for a toy car. Goal: smooth, safe city driving with active steering.

INPUT:
- position: left/center/right (center = in your path)
- distance_m: distance in meters
- current_speed: movement speed (0=stopped, higher=faster)
- movement_angle: CURRENT road direction (negative=left, positive=right, near zero=STRAIGHT)
- Count total objects to assess traffic density

===== STRAIGHT ROAD RULE (CRITICAL!) =====

When |movement_angle| < 5°: Road is STRAIGHT!
→ steering = "center"
→ steering_angle = 0
→ DO NOT keep previous turn direction!

This is the MOST IMPORTANT rule. Always return to center on straight roads.

===== ROAD FOLLOWING =====

The movement_angle shows the CURRENT road direction:

| movement_angle | Road State | Action |
|----------------|------------|--------|
| -5° to +5°     | STRAIGHT   | steering=center, angle=0 |
| -15° to -5°    | gentle left | steering=left, angle=10-15° |
| -30° to -15°   | left curve | steering=left, angle=15-30° |
| < -30°         | sharp left | steering=left, angle=30-50° |
| +5° to +15°    | gentle right | steering=right, angle=10-15° |
| +15° to +30°   | right curve | steering=right, angle=15-30° |
| > +30°         | sharp right | steering=right, angle=30-50° |

===== STEERING RULES =====

1. STRAIGHT ROAD (|movement_angle| < 5°):
   - ALWAYS: steering="center", steering_angle=0
   - Even if you just finished a turn!
   - Reset to center immediately when road straightens

2. CURVES (|movement_angle| >= 5°):
   - Follow the curve direction
   - steering_angle ≈ |movement_angle| × 1.0-1.2

3. OBSTACLE AVOIDANCE:
   - Object in CENTER < 5m → steer away if side clear
   - Add 10-20° to steering if avoiding obstacle
   - If BOTH sides blocked → brake, don't steer much

===== GEAR RULES =====

GEAR 1: Heavy traffic, obstacles < 5m, pedestrians, sharp curves (|angle|>30°)
GEAR 2: City driving, moderate traffic, gentle curves
GEAR 3: Light traffic, straight open road, nearest > 15m
GEAR 4-5: NEVER in city!

===== PEDAL RULES =====

1. CURVES:
   - |movement_angle| > 30° → gas 20-30%, gear 1-2
   - |movement_angle| > 45° → gas 15-25%, gear 1

2. PEDESTRIANS: ANY < 5m → gas 15-25%; CENTER < 3m → brake

3. VEHICLES: CENTER < 3m → brake; CENTER 3-15m → gas 25-40%

4. STRAIGHT + CLEAR: gas 40-50%

===== DECISION FLOW =====
1. Check |movement_angle|: if < 5° → steering=center, angle=0
2. If curve: steer in curve direction
3. If obstacle in path: adjust steering or brake
4. Select gear based on situation

RESPONSE (JSON only):
{"steering": "center", "steering_angle": 0, "pedal": "gas", "pedal_percent": 40, "gear": 2, "reasoning": "straight road, following traffic"}

EXAMPLES:
- movement_angle=2, car center 12m → center 0°, gas 35% (STRAIGHT!)
- movement_angle=-3, no objects → center 0°, gas 45% (STRAIGHT!)
- movement_angle=4, car left 5m → center 0°, gas 40% (STRAIGHT!)
- movement_angle=-18, no objects → left 20°, gas 35%
- movement_angle=25, car right 10m → right 28°, gas 30%
- movement_angle=-35, no objects → left 40°, gas 25%, gear 1"""

    def estimate_position(self, bbox: List[float], image_width: int) -> str:
        """
        Determines object position: left, center, right.
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_width: image width
        """
        center_x = (bbox[0] + bbox[2]) / 2
        relative_x = center_x / image_width
        
        if relative_x < 0.33:
            return "left"
        elif relative_x > 0.66:
            return "right"
        else:
            return "center"
    
    def estimate_distance(self, bbox: List[float], image_height: int) -> str:
        """
        Estimates distance to object by bbox size.
        Larger object (lower in frame) = closer.
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_height: image height
        """
        box_height = bbox[3] - bbox[1]
        relative_height = box_height / image_height
        
        # Also consider vertical position
        bottom_y = bbox[3] / image_height
        
        if relative_height > 0.3 or bottom_y > 0.8:
            return "close"
        elif relative_height > 0.15 or bottom_y > 0.6:
            return "medium"
        else:
            return "far"
    
    def estimate_distance_meters(self, bbox: List[float], image_height: int, class_name: str = "") -> float:
        """
        Estimates distance to object in meters based on bbox size and class.
        
        This is a heuristic approximation. For accurate distance:
        - Use stereo camera or depth sensor
        - Calibrate camera focal length and object real-world sizes
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_height: image height in pixels
            class_name: object class for size-aware estimation
        
        Returns:
            Estimated distance in meters
        """
        box_height = bbox[3] - bbox[1]
        relative_height = box_height / image_height
        
        # Typical object heights in meters (approximate)
        typical_heights = {
            "person": 1.7,
            "car": 1.5,
            "truck": 3.0,
            "bus": 3.2,
            "motorcycle": 1.2,
            "bicycle": 1.0,
            "traffic light": 0.6,
            "stop sign": 0.75,
        }
        
        # Get typical height for this object class
        object_height = typical_heights.get(class_name, 1.5)  # default 1.5m
        
        # Simplified pinhole camera model:
        # distance = (object_real_height * focal_length) / bbox_height
        # We use relative_height as a proxy for (bbox_height / image_height)
        # Assume ~60 degree vertical FOV (typical for dashcam)
        
        if relative_height < 0.01:
            return 50.0  # Very small object = very far
        
        # Calibration factor (adjust based on your camera)
        # Higher = further distances
        calibration_factor = 1.2
        
        # Distance estimation formula
        # When object fills 50% of frame height, it's approximately at 2m
        estimated_distance = (object_height * calibration_factor) / relative_height
        
        # Clamp to reasonable range
        estimated_distance = max(0.5, min(estimated_distance, 50.0))
        
        return round(estimated_distance, 1)
    
    def prepare_detections(
        self,
        boxes: List[List[float]],
        labels: List[int],
        scores: List[float],
        image_width: int,
        image_height: int,
        threshold: float = 0.5
    ) -> List[DetectedObject]:
        """
        Prepares list of detected objects.
        
        Args:
            boxes: list of bbox [[x1,y1,x2,y2], ...]
            labels: list of class_id
            scores: list of confidence scores
            image_width: image width
            image_height: image height
            threshold: minimum confidence threshold
        """
        detected = []
        
        for box, label, score in zip(boxes, labels, scores):
            if score < threshold:
                continue
            
            class_name = CLASSES[label] if 0 <= label < len(CLASSES) else f"unknown_{label}"
            
            if class_name == "__background__":
                continue
            
            obj = DetectedObject(
                class_name=class_name,
                confidence=round(score, 3),
                bbox=box,
                position=self.estimate_position(box, image_width),
                distance_estimate=self.estimate_distance(box, image_height),
                distance_meters=self.estimate_distance_meters(box, image_height, class_name),
            )
            detected.append(obj)
        
        return detected
    
    def generate_control_command(
        self,
        detected_objects: List[DetectedObject],
        current_speed: float = 0.0,
        movement_angle: float = 0.0,
        additional_context: str = "",
        **kwargs  # Accept extra kwargs for backwards compatibility
    ) -> ControlCommand:
        """
        Generates control command based on detected objects.
        
        Args:
            detected_objects: list of detected objects
            current_speed: current speed (if known)
            movement_angle: road direction from optical flow (negative=left, positive=right)
            additional_context: additional context
        
        Returns:
            ControlCommand with control commands
        """
        # Form scene description for LLM
        if not detected_objects:
            scene_description = "Road is clear, no objects detected."
            object_count = 0
            min_distance = 999
        else:
            objects_desc = []
            min_distance = 999
            for obj in detected_objects:
                desc = (
                    f"- {obj.class_name}: position={obj.position}, "
                    f"distance={obj.distance_estimate}, "
                    f"distance_m={obj.distance_meters}m, "
                    f"confidence={obj.confidence:.0%}"
                )
                objects_desc.append(desc)
                if obj.distance_meters < min_distance:
                    min_distance = obj.distance_meters
            object_count = len(detected_objects)
            scene_description = "Detected objects:\n" + "\n".join(objects_desc)
        
        # Describe road direction
        if abs(movement_angle) < 5:
            road_direction = "straight"
        elif movement_angle < -30:
            road_direction = "sharp left curve"
        elif movement_angle < -15:
            road_direction = "moderate left curve"
        elif movement_angle < 0:
            road_direction = "gentle left curve"
        elif movement_angle > 30:
            road_direction = "sharp right curve"
        elif movement_angle > 15:
            road_direction = "moderate right curve"
        else:
            road_direction = "gentle right curve"
        
        user_message = f"""Current situation:
{scene_description}

Road direction: movement_angle={movement_angle:.1f}° ({road_direction})
Traffic summary: {object_count} objects, nearest at {min_distance:.1f}m
Current speed: {current_speed:.1f}
{additional_context}

Generate control command. Remember: FOLLOW THE ROAD CURVE!"""

        # Debug: print what we send to LLM
        if hasattr(self, 'debug') and self.debug:
            print("\n" + "="*60)
            print("📤 SENDING TO LLM (TEXT ONLY, NO IMAGE!):")
            print("-"*60)
            print(user_message)
            print("="*60 + "\n")

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,  # Low temperature for stable responses
                max_tokens=200,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Debug: print LLM response
            if hasattr(self, 'debug') and self.debug:
                print("📥 LLM RESPONSE:")
                print(response_text)
                print("-"*60)
            
            # Parse JSON response
            # Remove possible markdown tags
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            command_dict = json.loads(response_text)
            
            return ControlCommand(
                steering=command_dict.get("steering", "center"),
                steering_angle=int(command_dict.get("steering_angle", 0)),
                pedal=command_dict.get("pedal", "brake"),
                pedal_percent=int(command_dict.get("pedal_percent", 0)),
                gear=int(command_dict.get("gear", 3)),
                reasoning=command_dict.get("reasoning", "")
            )
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"LLM response: {response_text}")
            # Return safe default command
            return ControlCommand(
                steering="center",
                steering_angle=0,
                pedal="brake",
                pedal_percent=50,
                gear=1,
                reasoning="Parsing error, safe stop"
            )
        except Exception as e:
            print(f"LLM call error: {e}")
            return ControlCommand(
                steering="center",
                steering_angle=0,
                pedal="brake",
                pedal_percent=100,
                gear=1,
                reasoning=f"LLM error: {e}, emergency stop"
            )
    
    def command_to_dict(self, command: ControlCommand) -> Dict:
        """Converts command to dictionary for JSON."""
        return {
            "steering": command.steering,
            "angle": f"{command.steering_angle} degrees",
            "pedal": command.pedal,
            "pedal_percent": f"{command.pedal_percent}%",
            "gear": command.gear,
            "reasoning": command.reasoning
        }


# Testing
if __name__ == "__main__":
    print("Testing LLM Controller")
    print("=" * 50)
    
    # Create test data
    test_objects = [
        DetectedObject(
            class_name="car",
            confidence=0.95,
            bbox=[200, 300, 400, 500],
            position="center",
            distance_estimate="medium"
        ),
        DetectedObject(
            class_name="pedestrian",
            confidence=0.87,
            bbox=[100, 350, 150, 450],
            position="left",
            distance_estimate="close"
        ),
    ]
    
    print("Test objects:")
    for obj in test_objects:
        print(f"  - {obj.class_name}: {obj.position}, {obj.distance_estimate}")
    
    print("\nAttempting connection to Azure OpenAI...")
    
    try:
        controller = LLMController()
        command = controller.generate_control_command(test_objects)
        
        print("\nGenerated command:")
        print(json.dumps(controller.command_to_dict(command), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure config.py has correct values for:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_DEPLOYMENT")
