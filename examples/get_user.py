"""
Example of getting current user information from the OpenElectricity API.

This example demonstrates how to:
1. Get the current user's information
2. Display user details including rate limits
"""

from openelectricity import OEClient


def main():
    """Get and display current user information."""
    with OEClient() as client:
        # Get current user info
        response = client.get_current_user()

        # Display user information
        user = response.data

        print("\nUser Information:")
        print(f"ID: {user.id}")
        print(f"Name: {user.full_name}")
        print(f"Email: {user.email}")
        print(f"Plan: {user.plan}")

        # Display rate limit information
        print("\nAPI Usage:")
        print(f"Remaining calls: {user.meta.remaining}")


if __name__ == "__main__":
    main()
