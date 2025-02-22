"""
Example of getting current user information from the OpenElectricity API.

This example demonstrates how to:
1. Get the current user's information
2. Display user details including rate limits

Required Environment Variables:
    OPENELECTRICITY_API_KEY: Your OpenElectricity API key
    OPENELECTRICITY_API_URL: (Optional) Override the default API URL
"""

import sys

from rich.console import Console

from openelectricity import OEClient
from openelectricity.settings_schema import settings


def main():
    """Get and display current user information."""
    console = Console()

    # Print settings for debugging
    console.print("\n[bold blue]API Settings:[/bold blue]")
    console.print(f"API URL: {settings.base_url}")
    console.print(f"Environment: {settings.env}")
    console.print(f"API Key: {settings.api_key[:8]}...")

    try:
        with OEClient() as client:
            # Get current user info
            console.print("\n[blue]Fetching user information...[/blue]")
            response = client.get_current_user()

            # Display user information
            user = response.data

            console.print("\n[green]User Information:[/green]")
            console.print(f"ID: {user.id}")
            console.print(f"Name: {user.full_name}")
            console.print(f"Email: {user.email}")
            console.print(f"Plan: {user.plan}")
            console.print(f"Roles: {', '.join(role.value for role in user.roles)}")

            # Display rate limit information if available
            if user.rate_limit:
                console.print("\n[yellow]Rate Limit Information:[/yellow]")
                console.print(f"Limit: {user.rate_limit.limit}")
                console.print(f"Remaining: {user.rate_limit.remaining}")
                console.print(f"Reset: {user.rate_limit.reset}")

            # Display API usage if available
            if user.meta:
                console.print("\n[blue]API Usage:[/blue]")
                console.print(f"Remaining calls: {user.meta.remaining}")
                if user.meta.reset:
                    console.print(f"Reset time: {user.meta.reset}")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
