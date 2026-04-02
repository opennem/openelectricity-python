"""
Example of getting current user information from the OpenElectricity API.

This example demonstrates how to:
1. Get the current user's information
2. Display user details including rate limits

Environment Variables:
    OPENELECTRICITY_API_KEY: Your OpenElectricity API key
    OPENELECTRICITY_API_URL: (Optional) Override the default API URL
"""

import sys

from dotenv import load_dotenv

from openelectricity import OEClient
from openelectricity.models.user import OpenNEMUser

load_dotenv()


def main() -> None:
    """Get and display current user information."""
    try:
        with OEClient() as client:
            response = client.get_current_user()

            user = response.data
            if not isinstance(user, OpenNEMUser):
                print("Invalid user data returned", file=sys.stderr)
                sys.exit(1)

            print(f"\nUser: {user.full_name} ({user.email})")  # type: ignore
            print(f"Plan: {user.plan}")  # type: ignore

            # Display rate limit information if available
            if user.rate_limit:  # type: ignore
                print(f"\nRate Limit: {user.rate_limit.remaining}/{user.rate_limit.limit} remaining")  # type: ignore

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
