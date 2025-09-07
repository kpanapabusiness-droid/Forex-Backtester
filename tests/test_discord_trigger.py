"""
Test file specifically designed to trigger Discord notification.
This test intentionally fails to validate CI Discord integration.
"""


def test_intentional_failure_for_discord():
    """
    This test intentionally fails to trigger the Discord notification system.
    It should be removed after validating the Discord integration works.
    """
    # This will cause pytest to fail and trigger Discord notification
    assert False, "Intentional failure to test Discord CI notification"


def test_another_failure_for_good_measure():
    """
    Additional failing test to ensure pytest exit code is non-zero.
    """
    raise ValueError("This is an intentional error to test Discord notifications")
