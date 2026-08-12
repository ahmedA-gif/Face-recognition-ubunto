"""Zone Engine for Polygon-Based Spatial Detection.

This module handles:
- Zone (polygon) management and intersection testing
- Occupancy counting per zone
- Zone entry/exit events
- Restricted zone intrusion detection
- Object left behind / removed detection (time-based)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any, Hashable
from collections import defaultdict

import numpy as np

from src.events.store import Event, EventsStore
from src.tracking.bytetrack import Track
from src.utils.geometry import (
    point_in_polygon,
    polygon_centroid,
    clean_polygon,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ZoneType(Enum):
    """Types of zones."""
    NORMAL = "normal"           # General zone (e.g., reception, lobby)
    RESTRICTED = "restricted"   # No entry allowed (e.g., server room)
    ENTRANCE = "entrance"       # Entry point
    EXIT = "exit"               # Exit point
    LINE = "line"               # Virtual line (for crossing detection)


@dataclass
class Zone:
    """A spatial zone defined by a polygon."""
    zone_id: str
    name: str
    polygon: List[Tuple[float, float]]  # Normalized coordinates [0, 1]
    zone_type: ZoneType = ZoneType.NORMAL
    
    # Configuration
    min_occupancy: Optional[int] = None  # For occupancy limit events
    max_occupancy: Optional[int] = None  # Maximum allowed people
    allowed_classes: List[str] = field(default_factory=lambda: ["person"])
    restricted: bool = False  # If zone is restricted
    
    # Runtime data (pixel coordinates)
    _pixel_polygon: Optional[List[Tuple[float, float]]] = None
    _centroid: Optional[Tuple[float, float]] = None
    
    def to_pixel(self, width: int, height: int) -> List[Tuple[float, float]]:
        """Convert normalized polygon to pixel coordinates."""
        if self._pixel_polygon is None or self._width != width or self._height != height:
            self._width = width
            self._height = height
            self._pixel_polygon = [
                (x * width, y * height) 
                for x, y in self.polygon
            ]
        return self._pixel_polygon
    
    def contains(self, point: Tuple[float, float], width: int, height: int) -> bool:
        """Check if point is inside the zone."""
        px = point[0] / width if width > 0 else 0
        py = point[1] / height if height > 0 else 0
        
        # Use normalized coordinates for testing
        return point_in_polygon((px, py), self.polygon)
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Get the centroid of the zone."""
        if self._centroid is None:
            self._centroid = polygon_centroid(self.polygon)
        return self._centroid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "polygon": self.polygon,
            "zone_type": self.zone_type.value,
            "restricted": self.restricted,
            "max_occupancy": self.max_occupancy,
            "allowed_classes": self.allowed_classes,
        }


@dataclass
class ZoneOccupancy:
    """Tracks occupancy of a zone."""
    zone_id: str
    current_count: int = 0
    person_tracks: Set[Hashable] = field(default_factory=set)
    last_entry_time: float = 0.0
    last_exit_time: float = 0.0
    occupancy_history: List[Tuple[float, int]] = field(default_factory=list)
    
    def add_person(self, track_key: Hashable, timestamp: float) -> None:
        """Add a person to the zone."""
        if track_key not in self.person_tracks:
            self.person_tracks.add(track_key)
            self.current_count += 1
            self.last_entry_time = timestamp
            self.occupancy_history.append((timestamp, self.current_count))
    
    def remove_person(self, track_key: Hashable, timestamp: float) -> None:
        """Remove a person from the zone."""
        if track_key in self.person_tracks:
            self.person_tracks.remove(track_key)
            self.current_count -= 1
            self.last_exit_time = timestamp
            self.occupancy_history.append((timestamp, self.current_count))
    
    def get_duration_above(self, threshold: int, timestamp: float) -> float:
        """Get total time occupancy was above threshold."""
        total_time = 0.0
        for i in range(1, len(self.occupancy_history)):
            t1, c1 = self.occupancy_history[i-1]
            t2, c2 = self.occupancy_history[i]
            if c1 >= threshold or c2 >= threshold:
                total_time += min(t2 - t1, timestamp - t1)
        return total_time


@dataclass
class ZoneEventConfig:
    """Configuration for zone-based events."""
    occupancy_limit_enabled: bool = True
    min_occupancy_duration: float = 5.0  # Seconds
    loitering_threshold: float = 30.0   # Seconds in zone = loitering
    restricted_zone_min_duration: float = 2.0  # Seconds for intrusion
    
    # For object left behind detection
    object_stationary_threshold: float = 20.0  # Seconds
    owner_distance_threshold: float = 5.0    # Meters


# =============================================================================
# ZONE ENGINE
# =============================================================================

class ZoneEngine:
    """Manages zones and generates zone-based events."""
    
    def __init__(
        self,
        camera_id: str = "cam_01",
        config: Optional[ZoneEventConfig] = None,
    ):
        self.camera_id = camera_id
        self.config = config or ZoneEventConfig()
        
        # Zone storage
        self._zones: Dict[str, Zone] = {}
        self._occupancy: Dict[str, ZoneOccupancy] = {}
        
        # Track to zone mapping
        self._track_zones: Dict[Hashable, Set[str]] = defaultdict(set)
        
        # Object tracking for left-behind detection
        self._objects: Dict[Hashable, Dict] = {}
        
        # Event counters
        self.counts: Dict[str, int] = defaultdict(int)
        
        # Frame dimensions (cached)
        self._frame_width: int = 0
        self._frame_height: int = 0
    
    # =========================================================================
    # ZONE MANAGEMENT
    # =========================================================================
    
    def add_zone(
        self,
        zone_id: str,
        name: str,
        polygon: List[Tuple[float, float]],
        zone_type: ZoneType = ZoneType.NORMAL,
        restricted: bool = False,
        max_occupancy: Optional[int] = None,
    ) -> None:
        """Add a new zone."""
        # Clean the polygon
        clean_poly = clean_polygon(polygon, max_points=12)
        
        self._zones[zone_id] = Zone(
            zone_id=zone_id,
            name=name,
            polygon=clean_poly,
            zone_type=zone_type,
            restricted=restricted,
            max_occupancy=max_occupancy,
        )
        self._occupancy[zone_id] = ZoneOccupancy(zone_id=zone_id)
        
        logger.info(f"Added zone: {zone_id} ({name})")
    
    def remove_zone(self, zone_id: str) -> bool:
        """Remove a zone."""
        if zone_id in self._zones:
            del self._zones[zone_id]
            del self._occupancy[zone_id]
            
            # Remove from track mappings
            for track_key, zones in list(self._track_zones.items()):
                zones.discard(zone_id)
                if not zones:
                    del self._track_zones[track_key]
            
            logger.info(f"Removed zone: {zone_id}")
            return True
        return False
    
    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get a zone by ID."""
        return self._zones.get(zone_id)
    
    def list_zones(self) -> List[str]:
        """List all zone IDs."""
        return list(self._zones.keys())
    
    def clear_zones(self) -> None:
        """Clear all zones."""
        self._zones.clear()
        self._occupancy.clear()
        self._track_zones.clear()
        logger.info("All zones cleared")
    
    # =========================================================================
    # TRACK PROCESSING
    # =========================================================================
    
    def _get_track_key(self, track: Track) -> Hashable:
        """Get a unique key for a track."""
        gid = track.meta.get("global_id") if track.meta else None
        if gid:
            return ("gid", gid)
        return ("tid", track.track_id)
    
    def _get_probe_point(self, track: Track) -> Tuple[float, float]:
        """Get the probe point (foot point by default)."""
        from src.utils.geometry import foot_point
        return foot_point(track.xyxy)
    
    def update(
        self,
        tracks: List[Track],
        frame_shape: Tuple[int, int, int],
        store: EventsStore,
    ) -> List[Event]:
        """Process tracks and generate zone-based events.
        
        Args:
            tracks: List of active tracks
            frame_shape: (height, width, channels)
            store: Event storage
            
        Returns:
            List of generated events
        """
        h, w = frame_shape[:2]
        self._frame_width = w
        self._frame_height = h
        now = time.time()
        
        produced_events: List[Event] = []
        alive_keys: set = set()
        
        # Process each track
        for track in tracks:
            key = self._get_track_key(track)
            alive_keys.add(key)
            
            # Get probe point
            point = self._get_probe_point(track)
            
            # Check which zones the track is in
            current_zones = set()
            for zone_id, zone in self._zones.items():
                if zone.contains(point, w, h):
                    current_zones.add(zone_id)
            
            # Get previous zones
            prev_zones = self._track_zones.get(key, set())
            
            # Detect zone entry
            new_zones = current_zones - prev_zones
            exited_zones = prev_zones - current_zones
            
            # Update track zones
            self._track_zones[key] = current_zones
            
            # Process zone entries
            for zone_id in new_zones:
                zone = self._zones[zone_id]
                
                # Update occupancy
                if zone_id in self._occupancy:
                    self._occupancy[zone_id].add_person(key, now)
                
                # Generate zone entry event
                if zone.zone_type == ZoneType.RESTRICTED:
                    # Restricted zone intrusion
                    event = self._create_zone_event(
                        track, zone_id, "intrusion", now, store
                    )
                    if event:
                        produced_events.append(event)
                        self.counts["restricted_intrusion"] += 1
                else:
                    # Normal zone entry
                    event = self._create_zone_event(
                        track, zone_id, "zone_entry", now, store
                    )
                    if event:
                        produced_events.append(event)
                        self.counts["zone_entry"] += 1
            
            # Process zone exits
            for zone_id in exited_zones:
                zone = self._zones[zone_id]
                
                # Update occupancy
                if zone_id in self._occupancy:
                    self._occupancy[zone_id].remove_person(key, now)
                
                # Generate zone exit event
                event = self._create_zone_event(
                    track, zone_id, "zone_exit", now, store
                )
                if event:
                    produced_events.append(event)
                    self.counts["zone_exit"] += 1
            
            # Check for occupancy limit violations
            self._check_occupancy_limits(now, store, produced_events)
            
            # Check for loitering
            self._check_loitering(key, current_zones, now, store, produced_events)
            
            # Update object tracking for left-behind detection
            self._update_object_tracking(track, current_zones, now)
        
        # Clean up removed tracks
        self._cleanup_removed_tracks(alive_keys, now)
        
        # Check for object left behind / removed
        self._check_object_events(now, store, produced_events)
        
        return produced_events
    
    def _create_zone_event(
        self,
        track: Track,
        zone_id: str,
        event_type: str,
        timestamp: float,
        store: EventsStore,
    ) -> Optional[Event]:
        """Create a zone-related event."""
        zone = self._zones.get(zone_id)
        if not zone:
            return None
        
        date_s, time_s = EventsStore.now_parts()
        person_name = track.person_name or f"Unknown#{track.track_id}"
        
        metadata = {
            "zone_id": zone_id,
            "zone_name": zone.name,
            "zone_type": zone.zone_type.value,
            "restricted": zone.restricted,
        }
        
        # Add occupancy info
        if zone_id in self._occupancy:
            occupancy = self._occupancy[zone_id]
            metadata["current_occupancy"] = occupancy.current_count
            if zone.max_occupancy:
                metadata["max_occupancy"] = zone.max_occupancy
        
        event = Event(
            date=date_s,
            time=time_s,
            person=person_name,
            direction=event_type,
            track_id=track.track_id,
            camera_id=self.camera_id,
            confidence=1.0,  # Zone events are deterministic
            metadata=metadata,
            event_id=str(uuid.uuid4()),
        )
        
        event.id = store.insert(event)
        logger.info(f"Generated {event_type} event in zone {zone_id}")
        
        return event
    
    # =========================================================================
    # OCCUPANCY MONITORING
    # =========================================================================
    
    def _check_occupancy_limits(
        self,
        now: float,
        store: EventsStore,
        events: List[Event],
    ) -> None:
        """Check for occupancy limit violations."""
        for zone_id, zone in self._zones.items():
            if zone.max_occupancy is None:
                continue
            
            occupancy = self._occupancy.get(zone_id)
            if not occupancy:
                continue
            
            if occupancy.current_count > zone.max_occupancy:
                # Check if this is a new violation
                # We need to track the start time of the violation
                start_time = now
                for t, count in reversed(occupancy.occupancy_history):
                    if count <= zone.max_occupancy:
                        start_time = t
                        break
                
                duration = now - start_time
                if duration >= self.config.min_occupancy_duration:
                    # Generate occupancy limit event
                    date_s, time_s = EventsStore.now_parts()
                    
                    event = Event(
                        date=date_s,
                        time=time_s,
                        person="multiple",
                        direction="occupancy_limit",
                        track_id=-1,  # Special value for zone-level events
                        camera_id=self.camera_id,
                        confidence=1.0,
                        metadata={
                            "zone_id": zone_id,
                            "zone_name": zone.name,
                            "current": occupancy.current_count,
                            "max": zone.max_occupancy,
                            "duration": duration,
                        },
                        event_id=str(uuid.uuid4()),
                    )
                    
                    event.id = store.insert(event)
                    events.append(event)
                    self.counts["occupancy_limit"] += 1
                    
                    logger.info(f"Occupancy limit exceeded in zone {zone_id}: {occupancy.current_count} > {zone.max_occupancy}")
    
    # =========================================================================
    # LOITERING DETECTION
    # =========================================================================
    
    def _check_loitering(
        self,
        track_key: Hashable,
        current_zones: Set[str],
        now: float,
        store: EventsStore,
        events: List[Event],
    ) -> None:
        """Check for loitering in restricted or monitored zones."""
        # Only check for person tracks
        if not current_zones:
            return
        
        # Get track's entry time for each zone
        for zone_id in current_zones:
            zone = self._zones.get(zone_id)
            if not zone:
                continue
            
            occupancy = self._occupancy.get(zone_id)
            if not occupancy:
                continue
            
            # Check if this track has been in the zone for too long
            if track_key in occupancy.person_tracks:
                # Find when this track entered
                entry_time = None
                for t, count in reversed(occupancy.occupancy_history):
                    # This is simplified; in practice we'd need to track per-track entry time
                    pass
                
                # For now, use a simplified approach
                # We'll track entry times separately
                if zone_id not in self._zone_entry_times:
                    self._zone_entry_times = {}
                
                if track_key not in self._zone_entry_times.get(zone_id, {}):
                    continue
                
                entry_time = self._zone_entry_times[zone_id][track_key]
                duration = now - entry_time
                
                if duration >= self.config.loitering_threshold:
                    # Generate loitering event
                    date_s, time_s = EventsStore.now_parts()
                    
                    event = Event(
                        date=date_s,
                        time=time_s,
                        person=f"Unknown#{track_key[1]}" if isinstance(track_key, tuple) else "Unknown",
                        direction="loitering",
                        track_id=track_key[1] if isinstance(track_key, tuple) else -1,
                        camera_id=self.camera_id,
                        confidence=min(duration / self.config.loitering_threshold, 1.0),
                        metadata={
                            "zone_id": zone_id,
                            "zone_name": zone.name,
                            "duration": duration,
                        },
                        event_id=str(uuid.uuid4()),
                    )
                    
                    event.id = store.insert(event)
                    events.append(event)
                    self.counts["loitering"] += 1
                    
                    # Reset entry time to prevent repeated events
                    del self._zone_entry_times[zone_id][track_key]
                    
                    logger.info(f"Loitering detected in zone {zone_id} (duration: {duration:.1f}s)")
    
    # =========================================================================
    # OBJECT TRACKING (for left-behind detection)
    # =========================================================================
    
    def _update_object_tracking(
        self,
        track: Track,
        current_zones: Set[str],
        now: float,
    ) -> None:
        """Track objects for left-behind detection."""
        key = self._get_track_key(track)
        
        # Only track non-person objects for now
        # In practice, you'd filter by object class
        obj_class = track.meta.get("class") if track.meta else "person"
        
        if obj_class == "person":
            # Track entry times for loitering
            for zone_id in current_zones:
                if zone_id not in self._zone_entry_times:
                    self._zone_entry_times[zone_id] = {}
                if key not in self._zone_entry_times[zone_id]:
                    self._zone_entry_times[zone_id][key] = now
        else:
            # Track object state
            if key not in self._objects:
                self._objects[key] = {
                    "class": obj_class,
                    "first_seen": now,
                    "last_seen": now,
                    "last_zone": None,
                    "stationary_since": now,
                    "owner_track": None,
                }
            
            obj = self._objects[key]
            obj["last_seen"] = now
            
            # Update zone
            if current_zones:
                obj["last_zone"] = list(current_zones)[0]
            
            # Check if stationary
            point = self._get_probe_point(track)
            if "last_position" in obj:
                prev_point = obj["last_position"]
                distance = ((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)**0.5
                if distance < 5.0:  # Less than 5px movement
                    if "stationary_start" not in obj:
                        obj["stationary_start"] = now
                else:
                    if "stationary_start" in obj:
                        del obj["stationary_start"]
            
            obj["last_position"] = point
    
    def _check_object_events(
        self,
        now: float,
        store: EventsStore,
        events: List[Event],
    ) -> None:
        """Check for object left-behind or removed events."""
        # Check for left-behind objects
        for key, obj in list(self._objects.items()):
            if "stationary_start" in obj:
                stationary_duration = now - obj["stationary_start"]
                
                if stationary_duration >= self.config.object_stationary_threshold:
                    # Check if owner is far away
                    # This would require tracking the person who was with the object
                    
                    # For now, just generate the event
                    date_s, time_s = EventsStore.now_parts()
                    
                    event = Event(
                        date=date_s,
                        time=time_s,
                        person="object",
                        direction="left_behind",
                        track_id=key[1] if isinstance(key, tuple) else -1,
                        camera_id=self.camera_id,
                        confidence=min(stationary_duration / self.config.object_stationary_threshold, 1.0),
                        metadata={
                            "object_class": obj.get("class", "unknown"),
                            "zone_id": obj.get("last_zone"),
                            "duration": stationary_duration,
                        },
                        event_id=str(uuid.uuid4()),
                    )
                    
                    event.id = store.insert(event)
                    events.append(event)
                    self.counts["object_left_behind"] += 1
                    
                    # Remove from tracking
                    del self._objects[key]
                    
                    logger.info(f"Object left behind: {obj.get('class')} in zone {obj.get('last_zone')}")
        
        # Check for removed objects (objects that disappeared)
        # This would require tracking objects that are no longer seen
        # Implementation left as exercise
    
    def _cleanup_removed_tracks(self, alive_keys: set, now: float) -> None:
        """Clean up removed tracks."""
        for key in [k for k in self._track_zones if k not in alive_keys]:
            # Remove from zone occupancy
            for zone_id in self._track_zones.get(key, set()):
                if zone_id in self._occupancy:
                    self._occupancy[zone_id].remove_person(key, now)
            
            # Remove from zone entry times
            if hasattr(self, '_zone_entry_times'):
                for zone_id in self._zone_entry_times:
                    if key in self._zone_entry_times[zone_id]:
                        del self._zone_entry_times[zone_id][key]
            
            del self._track_zones[key]
    
    # =============================================================================
    # LINE CROSSING (for backward compatibility)
    # =============================================================================
    
    def add_line(
        self,
        line_id: str,
        start: Tuple[float, float],
        end: Tuple[float, float],
        allowed_directions: Optional[List[str]] = None,
    ) -> None:
        """Add a virtual line for crossing detection."""
        # Lines are stored as degenerate polygons (2-point polygons)
        polygon = [start, end]
        self.add_zone(
            zone_id=line_id,
            name=f"line_{line_id}",
            polygon=polygon,
            zone_type=ZoneType.LINE,
        )
        
        # Store direction constraints
        if allowed_directions:
            self._zones[line_id].allowed_classes = allowed_directions
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def reset(self) -> None:
        """Reset all state."""
        self._zones.clear()
        self._occupancy.clear()
        self._track_zones.clear()
        self._objects.clear()
        self.counts.clear()
        logger.info("ZoneEngine reset")
    
    def get_occupancy(self, zone_id: str) -> int:
        """Get current occupancy of a zone."""
        occupancy = self._occupancy.get(zone_id)
        return occupancy.current_count if occupancy else 0
    
    def get_all_occupancy(self) -> Dict[str, int]:
        """Get occupancy for all zones."""
        return {
            zone_id: occ.current_count 
            for zone_id, occ in self._occupancy.items()
        }
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "zones": len(self._zones),
            "total_events": sum(self.counts.values()),
            "counts": dict(self.counts),
            "occupancy": self.get_all_occupancy(),
        }


# Import uuid for event_id generation
import uuid
