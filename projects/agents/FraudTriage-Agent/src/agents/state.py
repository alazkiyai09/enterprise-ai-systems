"""
Agent state definition for LangGraph workflow.

This module re-exports the canonical FraudTriageState from src.models.state
and provides a compatibility layer for create_initial_state.
"""

from datetime import datetime
from typing import Any

from langchain_core.messages import BaseMessage

# Import the canonical state definition from models.state
from src.models.state import (
    AlertType,
    FraudTriageState,
    RiskLevel,
    create_initial_state as create_state_from_models,
)

# Re-export FraudTriageState as AgentState for backward compatibility
AgentState = FraudTriageState


def create_initial_state(
    alert_id: str | dict[str, Any],
    alert_data: dict[str, Any] | None = None,
    alert_type: AlertType | str = AlertType.OTHER,
    transaction_amount: float = 0.0,
    customer_id: str = "",
    **kwargs: Any,
) -> FraudTriageState:
    """
    Create initial fraud triage state from alert data.

    This is a compatibility wrapper that converts from the legacy alert_data dict
    format to the new structured FraudTriageState.

    Supports two calling patterns:
    1. Legacy: create_initial_state(alert_data_dict) - single dict argument
    2. New: create_initial_state(alert_id, alert_data, ...) - named arguments

    Args:
        alert_id: Unique alert identifier OR legacy alert_data dict
        alert_data: Raw alert data (legacy format, optional)
        alert_type: Type of fraud alert (defaults to OTHER if not specified)
        transaction_amount: Amount of flagged transaction
        customer_id: Customer identifier
        **kwargs: Additional optional fields

    Returns:
        Initial state dictionary with required fields populated
    """
    # Handle legacy calling pattern: create_initial_state(alert_data_dict)
    if isinstance(alert_id, dict):
        alert_data = alert_id
        alert_id = alert_data.get("alert_id") or alert_data.get("id") or "unknown"

    # If alert_data dict is provided, extract values from it
    if alert_data:
        alert_id = alert_data.get("alert_id") or alert_data.get("id") or alert_id
        customer_id = customer_id or alert_data.get("customer_id", "")
        transaction_amount = float(
            alert_data.get("transaction", {}).get("amount", transaction_amount)
        )
        alert_type_str = alert_data.get("alert_type", alert_type)
        if isinstance(alert_type_str, str):
            try:
                alert_type = AlertType(alert_type_str)
            except ValueError:
                alert_type = AlertType.OTHER

        # Map alert_data fields to state fields
        kwargs.setdefault("account_id", alert_data.get("account_id"))
        kwargs.setdefault("transaction_id", alert_data.get("transaction", {}).get("transaction_id"))
        kwargs.setdefault("transaction_country", alert_data.get("transaction", {}).get("location_country"))
        kwargs.setdefault("transaction_device_id", alert_data.get("transaction", {}).get("device_id"))
        kwargs.setdefault("transaction_ip", alert_data.get("transaction", {}).get("ip_address"))
        kwargs.setdefault("rule_id", alert_data.get("rule_id"))
        kwargs.setdefault("alert_reason", alert_data.get("alert_reason"))
        # Store alert_data for backward compatibility
        kwargs.setdefault("alert_data", alert_data)

    return create_state_from_models(
        alert_id=alert_id,
        alert_type=alert_type,
        transaction_amount=transaction_amount,
        customer_id=customer_id,
        **kwargs,
    )


def state_to_dict(state: FraudTriageState) -> dict[str, Any]:
    """
    Convert fraud triage state to a regular dictionary for serialization.

    Args:
        state: Fraud triage state

    Returns:
        Dictionary representation of state
    """
    result = {}

    for key, value in state.items():
        if isinstance(value, list) and value and isinstance(value[0], BaseMessage):
            # Convert BaseMessage list to dict
            result[key] = [msg.dict() for msg in value]
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        else:
            result[key] = value

    return result


# Re-export for backward compatibility
__all__ = [
    "FraudTriageState",
    "AgentState",  # Alias for backward compatibility
    "create_initial_state",
    "state_to_dict",
    "AlertType",
    "RiskLevel",
]
