"""Rule Engine for Category 1 Events.

This module provides a configurable rule-based system for:
- Enabling/disabling specific events
- Setting thresholds and parameters per zone/camera
- Filtering events based on conditions
- Routing events to different outputs

Rules are defined in YAML and support:
- Simple boolean conditions
- Numeric comparisons
- String matching
- Logical AND/OR/NOT combinations
- Time-based conditions
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import re

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# RULE DEFINITIONS
# =============================================================================

class ConditionType(Enum):
    """Types of conditions."""
    EQUAL = "eq"
    NOT_EQUAL = "neq"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    NOT_REGEX = "not_regex"
    BETWEEN = "between"
    NOT_BETWEEN = "not_between"
    ALWAYS_TRUE = "always_true"
    ALWAYS_FALSE = "always_false"


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class Condition:
    """A single condition for event filtering."""
    field: str                    # Field to check (e.g., "confidence", "zone_id")
    operator: ConditionType       # Comparison operator
    value: Any                    # Value to compare against
    
    def evaluate(self, event: Dict[str, Any]) -> bool:
        """Evaluate this condition against an event."""
        actual = self._get_field_value(event)
        
        try:
            if self.operator == ConditionType.ALWAYS_TRUE:
                return True
            elif self.operator == ConditionType.ALWAYS_FALSE:
                return False
            elif self.operator == ConditionType.EQUAL:
                return actual == self.value
            elif self.operator == ConditionType.NOT_EQUAL:
                return actual != self.value
            elif self.operator == ConditionType.GREATER_THAN:
                return float(actual) > float(self.value)
            elif self.operator == ConditionType.GREATER_THAN_OR_EQUAL:
                return float(actual) >= float(self.value)
            elif self.operator == ConditionType.LESS_THAN:
                return float(actual) < float(self.value)
            elif self.operator == ConditionType.LESS_THAN_OR_EQUAL:
                return float(actual) <= float(self.value)
            elif self.operator == ConditionType.IN:
                return actual in self.value
            elif self.operator == ConditionType.NOT_IN:
                return actual not in self.value
            elif self.operator == ConditionType.CONTAINS:
                return self.value in str(actual)
            elif self.operator == ConditionType.NOT_CONTAINS:
                return self.value not in str(actual)
            elif self.operator == ConditionType.REGEX:
                return bool(re.search(str(self.value), str(actual)))
            elif self.operator == ConditionType.NOT_REGEX:
                return not re.search(str(self.value), str(actual))
            elif self.operator == ConditionType.BETWEEN:
                val = float(actual)
                min_val, max_val = self.value
                return min_val <= val <= max_val
            elif self.operator == ConditionType.NOT_BETWEEN:
                val = float(actual)
                min_val, max_val = self.value
                return not (min_val <= val <= max_val)
        except (ValueError, TypeError) as e:
            logger.warning(f"Condition evaluation error: {e}")
            return False
    
    def _get_field_value(self, event: Dict[str, Any]) -> Any:
        """Get the value of a field from an event (supports nested fields)."""
        parts = self.field.split(".")
        value = event
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        
        return value


@dataclass
class ConditionGroup:
    """A group of conditions combined with logical operators."""
    operator: LogicalOperator = LogicalOperator.AND
    conditions: List[Union[Condition, ConditionGroup]] = field(default_factory=list)
    
    def evaluate(self, event: Dict[str, Any]) -> bool:
        """Evaluate this condition group."""
        if self.operator == LogicalOperator.AND:
            return all(
                cond.evaluate(event) 
                for cond in self.conditions
            )
        elif self.operator == LogicalOperator.OR:
            return any(
                cond.evaluate(event) 
                for cond in self.conditions
            )
        elif self.operator == LogicalOperator.NOT:
            if len(self.conditions) != 1:
                logger.warning("NOT operator should have exactly one condition")
                return False
            return not self.conditions[0].evaluate(event)
        
        return False


@dataclass
class Action:
    """An action to take when a rule matches."""
    action_type: str           # e.g., "notify", "webhook", "log", "reject"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, event: Dict[str, Any]) -> bool:
        """Execute this action.
        
        Returns: True if action succeeded, False otherwise
        """
        if self.action_type == "log":
            logger.info(f"Rule action: {self.parameters.get('message', 'Event matched')}")
            return True
        elif self.action_type == "notify":
            # Send notification (implementation depends on system)
            return self._send_notification(event)
        elif self.action_type == "webhook":
            return self._send_webhook(event)
        elif self.action_type == "reject":
            # Mark event for rejection
            event["_rejected"] = True
            event["_reject_reason"] = self.parameters.get("reason", "Rule rejection")
            return True
        elif self.action_type == "tag":
            # Add tags to event
            tags = self.parameters.get("tags", [])
            if "_tags" not in event:
                event["_tags"] = []
            event["_tags"].extend(tags)
            return True
        elif self.action_type == "set_field":
            # Set a field on the event
            field_name = self.parameters.get("field")
            field_value = self.parameters.get("value")
            if field_name:
                event[field_name] = field_value
            return True
        
        return False
    
    def _send_notification(self, event: Dict[str, Any]) -> bool:
        """Send a notification."""
        # Implementation depends on notification system
        # For now, just log
        logger.info(f"Notification: {self.parameters.get('message', 'Event notification')}")
        return True
    
    def _send_webhook(self, event: Dict[str, Any]) -> bool:
        """Send a webhook."""
        import requests
        
        url = self.parameters.get("url")
        if not url:
            logger.error("Webhook URL not configured")
            return False
        
        payload = self.parameters.get("payload", event)
        headers = self.parameters.get("headers", {"Content-Type": "application/json"})
        method = self.parameters.get("method", "POST")
        
        try:
            response = requests.request(
                method=method,
                url=url,
                json=payload,
                headers=headers,
                timeout=5,
            )
            return response.status_code in (200, 201, 204)
        except Exception as e:
            logger.error(f"Webhook failed: {e}")
            return False


@dataclass
class Rule:
    """A rule for event processing."""
    rule_id: str
    name: str
    description: str = ""
    event_types: List[str] = field(default_factory=list)  # Events this rule applies to
    conditions: Union[Condition, ConditionGroup] = field(default_factory=ConditionGroup)
    actions: List[Action] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    
    def matches(self, event: Dict[str, Any]) -> bool:
        """Check if this rule matches the event."""
        if not self.enabled:
            return False
        
        # Check event type
        event_type = event.get("direction") or event.get("event_type") or event.get("_event_type")
        if self.event_types and event_type not in self.event_types:
            return False
        
        # Check conditions
        return self.conditions.evaluate(event)
    
    def execute(self, event: Dict[str, Any]) -> bool:
        """Execute all actions for this rule."""
        if not self.matches(event):
            return False
        
        for action in self.actions:
            if not action.execute(event):
                logger.warning(f"Action failed in rule {self.rule_id}")
        
        return True


@dataclass
class RuleEngineConfig:
    """Configuration for the rule engine."""
    rules_path: str = "config/category1_rules.yaml"
    max_rules: int = 100
    check_interval: float = 1.0  # Seconds between rule reloading
    auto_reload: bool = True


# =============================================================================
# RULE ENGINE
# =============================================================================

class RuleEngine:
    """Rule-based engine for filtering and processing Category 1 events."""
    
    def __init__(self, config: Optional[RuleEngineConfig] = None):
        self.config = config or RuleEngineConfig()
        self._rules: Dict[str, Rule] = {}
        self._last_load_time: float = 0.0
        self._last_check_time: float = 0.0
        
        # Load rules
        self.load_rules()
    
    def load_rules(self) -> bool:
        """Load rules from YAML file."""
        rules_path = Path(self.config.rules_path)
        
        if not rules_path.exists():
            logger.warning(f"Rules file not found: {rules_path}")
            return False
        
        try:
            with open(rules_path) as f:
                rules_data = yaml.safe_load(f)
            
            self._rules = {}
            
            for rule_data in rules_data.get("rules", []):
                rule = self._parse_rule(rule_data)
                if rule:
                    self._rules[rule.rule_id] = rule
            
            self._last_load_time = time.time()
            logger.info(f"Loaded {len(self._rules)} rules from {rules_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
            return False
    
    def _parse_rule(self, rule_data: Dict[str, Any]) -> Optional[Rule]:
        """Parse a rule from YAML data."""
        try:
            conditions = self._parse_conditions(rule_data.get("conditions", {}))
            actions = self._parse_actions(rule_data.get("actions", []))
            
            return Rule(
                rule_id=rule_data.get("rule_id", ""),
                name=rule_data.get("name", ""),
                description=rule_data.get("description", ""),
                event_types=rule_data.get("event_types", []),
                conditions=conditions,
                actions=actions,
                enabled=rule_data.get("enabled", True),
                priority=rule_data.get("priority", 0),
            )
        except Exception as e:
            logger.error(f"Error parsing rule {rule_data.get('rule_id')}: {e}")
            return None
    
    def _parse_conditions(self, conditions_data: Any) -> ConditionGroup:
        """Parse conditions from YAML data."""
        if isinstance(conditions_data, dict):
            # Single condition or grouped conditions
            if "and" in conditions_data:
                return ConditionGroup(
                    operator=LogicalOperator.AND,
                    conditions=[
                        self._parse_conditions(c) 
                        for c in conditions_data["and"]
                    ]
                )
            elif "or" in conditions_data:
                return ConditionGroup(
                    operator=LogicalOperator.OR,
                    conditions=[
                        self._parse_conditions(c) 
                        for c in conditions_data["or"]
                    ]
                )
            elif "not" in conditions_data:
                return ConditionGroup(
                    operator=LogicalOperator.NOT,
                    conditions=[self._parse_conditions(conditions_data["not"])]
                )
            else:
                # Single condition
                return self._parse_single_condition(conditions_data)
        elif isinstance(conditions_data, list):
            # List of conditions (AND by default)
            return ConditionGroup(
                operator=LogicalOperator.AND,
                conditions=[
                    self._parse_single_condition(c) 
                    for c in conditions_data
                ]
            )
        
        # Default: always true
        return Condition(
            field="_dummy",
            operator=ConditionType.ALWAYS_TRUE,
            value=True,
        )
    
    def _parse_single_condition(self, condition_data: Dict[str, Any]) -> Condition:
        """Parse a single condition."""
        if not isinstance(condition_data, dict):
            return Condition(
                field="_dummy",
                operator=ConditionType.ALWAYS_TRUE,
                value=True,
            )
        
        # Get operator
        for op_name, op_type in ConditionType.__members__.items():
            if op_name.lower() in condition_data:
                field = condition_data.get("field", "")
                value = condition_data[op_name.lower()]
                return Condition(
                    field=field,
                    operator=op_type,
                    value=value,
                )
        
        # Default: check for "field", "op", "value" format
        return Condition(
            field=condition_data.get("field", ""),
            operator=ConditionType(condition_data.get("op", "eq")),
            value=condition_data.get("value"),
        )
    
    def _parse_actions(self, actions_data: List[Dict[str, Any]]) -> List[Action]:
        """Parse actions from YAML data."""
        actions = []
        
        for action_data in actions_data:
            action = Action(
                action_type=action_data.get("type", "log"),
                parameters=action_data.get("parameters", action_data),
            )
            # Remove 'type' from parameters
            action.parameters.pop("type", None)
            actions.append(action)
        
        return actions
    
    def check_auto_reload(self) -> bool:
        """Check if rules should be reloaded."""
        if not self.config.auto_reload:
            return False
        
        now = time.time()
        if now - self._last_check_time >= self.config.check_interval:
            self._last_check_time = now
            rules_path = Path(self.config.rules_path)
            
            if rules_path.exists():
                mtime = rules_path.stat().st_mtime
                if mtime > self._last_load_time:
                    logger.info("Rules file changed, reloading...")
                    return self.load_rules()
        
        return False
    
    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process an event through all rules.
        
        Returns: The event (potentially modified or marked for rejection)
        """
        # Check for auto-reload
        self.check_auto_reload()
        
        # Convert event to dict if it's an Event object
        if hasattr(event, "to_dict"):
            event_dict = event.to_dict()
        else:
            event_dict = dict(event)
        
        # Apply rules
        for rule in sorted(self._rules.values(), key=lambda r: r.priority, reverse=True):
            rule.execute(event_dict)
        
        # Convert back if needed
        if hasattr(event, "from_dict"):
            return event.from_dict(event_dict)
        
        return event_dict
    
    def process_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple events through all rules."""
        return [self.process_event(e) for e in events]
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)
    
    def list_rules(self) -> List[str]:
        """List all rule IDs."""
        return list(self._rules.keys())
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule."""
        self._rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False
    
    def enable_rule(self, rule_id: str, enabled: bool) -> bool:
        """Enable or disable a rule."""
        rule = self._rules.get(rule_id)
        if rule:
            rule.enabled = enabled
            return True
        return False


# =============================================================================
# DEFAULT RULES FOR CATEGORY 1
# =============================================================================

DEFAULT_RULES_YAML = """
# Default Rules for Category 1 Events

rules:
  # High confidence events - always process
  - rule_id: "high_confidence_passthrough"
    name: "High Confidence Passthrough"
    description: "Allow events with confidence >= 0.9 to pass through without filtering"
    event_types: ["person_entered", "person_exited", "vehicle_entered", "restricted_zone_intrusion"]
    conditions:
      - field: "confidence"
        gte: 0.9
    actions:
      - type: "tag"
        parameters:
          tags: ["high_confidence"]
    enabled: true
    priority: 10

  # Low confidence events - require review
  - rule_id: "low_confidence_review"
    name: "Low Confidence Review"
    description: "Flag events with confidence < 0.8 for review"
    event_types: ["person_entered", "person_exited"]
    conditions:
      - field: "confidence"
        lt: 0.8
    actions:
      - type: "tag"
        parameters:
          tags: ["needs_review"]
      - type: "set_field"
        parameters:
          field: "_review_required"
          value: true
    enabled: true
    priority: 5

  # Restricted zone events - high priority
  - rule_id: "restricted_zone_alert"
    name: "Restricted Zone Alert"
    description: "Send immediate notification for restricted zone intrusions"
    event_types: ["restricted_zone_intrusion"]
    conditions:
      or:
        - field: "zone_id"
          in: ["server_room", "vault", "control_room"]
        - field: "metadata.zone_id"
          in: ["server_room", "vault", "control_room"]
    actions:
      - type: "notify"
        parameters:
          message: "RESTRICTED ZONE INTRUSION: {{person}} in {{zone_name}}"
      - type: "tag"
        parameters:
          tags: ["security_alert", "restricted_zone"]
    enabled: true
    priority: 100

  # Occupancy limit events
  - rule_id: "occupancy_alert"
    name: "Occupancy Limit Alert"
    description: "Alert when occupancy exceeds limit"
    event_types: ["occupancy_limit"]
    conditions:
      - field: "metadata.current"
        gt: 10
    actions:
      - type: "notify"
        parameters:
          message: "OCCUPANCY LIMIT EXCEEDED: {{metadata.current}} people in {{metadata.zone_name}}"
      - type: "tag"
        parameters:
          tags: ["occupancy_alert"]
    enabled: true
    priority: 20

  # Vehicle events - add to vehicle log
  - rule_id: "vehicle_logging"
    name: "Vehicle Event Logging"
    description: "Log all vehicle events to vehicle-specific log"
    event_types: ["vehicle_entered", "vehicle_exited"]
    actions:
      - type: "log"
        parameters:
          message: "VEHICLE EVENT: {{direction}} - {{license_plate}}"
      - type: "tag"
        parameters:
          tags: ["vehicle"]
    enabled: true
    priority: 1

  # Loitering events
  - rule_id: "loitering_alert"
    name: "Loitering Alert"
    description: "Alert on loitering events in sensitive areas"
    event_types: ["loitering"]
    conditions:
      and:
        - field: "metadata.duration"
          gte: 60.0
        - field: "metadata.zone_id"
          in: ["parking", "entrance", "lobby"]
    actions:
      - type: "notify"
        parameters:
          message: "LOITERING ALERT: Person in {{metadata.zone_name}} for {{metadata.duration}}s"
      - type: "tag"
        parameters:
          tags: ["security_alert", "loitering"]
    enabled: true
    priority: 30
"""


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global rule engine instance
_rule_engine: Optional[RuleEngine] = None


def get_rule_engine() -> RuleEngine:
    """Get or create the global rule engine."""
    global _rule_engine
    if _rule_engine is None:
        _rule_engine = RuleEngine()
    return _rule_engine


def load_default_rules() -> None:
    """Load default rules."""
    import tempfile
    
    # Create temp file with default rules
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(DEFAULT_RULES_YAML)
        temp_path = f.name
    
    try:
        engine = get_rule_engine()
        engine.config = RuleEngineConfig(rules_path=temp_path)
        engine.load_rules()
    finally:
        # Clean up temp file (optional - could keep it)
        pass


# =============================================================================
# RULE-BASED EVENT FILTER FOR CATEGORY 1
# =============================================================================

class Category1EventFilter:
    """Filters Category 1 events based on rules."""
    
    def __init__(self, rule_engine: Optional[RuleEngine] = None):
        self.rule_engine = rule_engine or get_rule_engine()
        self._rejected_events: List[Dict[str, Any]] = []
    
    def filter_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Filter an event based on rules.
        
        Returns: The event if it should be kept, None if rejected
        """
        processed = self.rule_engine.process_event(event)
        
        if processed.get("_rejected"):
            self._rejected_events.append(processed)
            logger.debug(f"Event rejected by rule: {processed.get('_reject_reason')}")
            return None
        
        return processed
    
    def filter_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter multiple events."""
        return [
            e for e in (self.filter_event(e) for e in events) 
            if e is not None
        ]
    
    def get_rejected_events(self) -> List[Dict[str, Any]]:
        """Get list of rejected events."""
        return self._rejected_events
    
    def clear_rejected_events(self) -> None:
        """Clear rejected events list."""
        self._rejected_events.clear()
