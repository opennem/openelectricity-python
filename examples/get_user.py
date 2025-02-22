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
        print(f"Roles: {', '.join(role.value for role in user.roles)}")

        # Display rate limit information if available
        if user.rate_limit:
            print("\nRate Limit Information:")
            print(f"Limit: {user.rate_limit.limit}")
            print(f"Remaining: {user.rate_limit.remaining}")
            print(f"Reset: {user.rate_limit.reset}")

        # Display API usage if available
        if user.meta:
            print("\nAPI Usage:")
            print(f"Remaining calls: {user.meta.remaining}")
            if user.meta.reset:
                print(f"Reset time: {user.meta.reset}")


if __name__ == "__main__":
    main()
