#!/usr/bin/env python3
"""
Script to upload a wheel file to a Unity Catalog volume using the Databricks SDK.
"""

import os
import argparse
import sys
import io

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_volume_exists(w, catalog_name, schema_name, volume_name):
    """
    Ensure that the catalog, schema, and volume exist. Create them if they don't.
    
    Args:
        w: WorkspaceClient instance
        catalog_name (str): The Unity Catalog catalog name
        schema_name (str): The schema name within the catalog
        volume_name (str): The volume name within the schema
    """
    # Check/create catalog
    try:
        w.catalogs.get(catalog_name)
        logger.info(f"Catalog '{catalog_name}' exists")
    except Exception:
        logger.info(f"Catalog '{catalog_name}' not found, creating...")
        try:
            w.catalogs.create(name=catalog_name)
            logger.info(f"Created catalog '{catalog_name}'")
        except Exception as e:
            logger.warning(f"Could not create catalog: {str(e)}")
    
    # Check/create schema
    try:
        w.schemas.get(f"{catalog_name}.{schema_name}")
        logger.info(f"Schema '{catalog_name}.{schema_name}' exists")
    except Exception:
        logger.info(f"Schema '{catalog_name}.{schema_name}' not found, creating...")
        try:
            w.schemas.create(name=schema_name, catalog_name=catalog_name)
            logger.info(f"Created schema '{catalog_name}.{schema_name}'")
        except Exception as e:
            logger.warning(f"Could not create schema: {str(e)}")
    
    # Check/create volume
    try:
        w.volumes.read(f"{catalog_name}.{schema_name}.{volume_name}")
        logger.info(f"Volume '{catalog_name}.{schema_name}.{volume_name}' exists")
    except Exception:
        logger.info(f"Volume '{catalog_name}.{schema_name}.{volume_name}' not found, creating...")
        try:
            w.volumes.create(
                catalog_name=catalog_name,
                schema_name=schema_name,
                name=volume_name,
                volume_type=VolumeType.MANAGED
            )
            logger.info(f"Created volume '{catalog_name}.{schema_name}.{volume_name}'")
        except Exception as e:
            logger.error(f"Could not create volume: {str(e)}")
            raise

def upload_wheel_to_volume(catalog_name, schema_name, volume_name, wheel_file_path):
    """
    Upload a wheel file to a Unity Catalog volume.
    
    Args:
        catalog_name (str): The Unity Catalog catalog name
        schema_name (str): The schema name within the catalog
        volume_name (str): The volume name within the schema
        wheel_file_path (str): Path to the wheel file to upload
    """
    try:
        # Initialize the Databricks SDK client
        w = WorkspaceClient()
        
        # Log workspace information
        workspace_url = w.config.host
        logger.info(f"Connected to Databricks workspace: {workspace_url}")
        
        # Check if wheel file exists
        if not os.path.exists(wheel_file_path):
            raise FileNotFoundError(f"Wheel file not found: {wheel_file_path}")
        
        # Ensure volume exists
        ensure_volume_exists(w, catalog_name, schema_name, volume_name)
        
        logger.info(f"Starting upload of wheel file: {wheel_file_path}")
        
        # Read file into bytes
        with open(wheel_file_path, "rb") as f:
            file_bytes = f.read()
        binary_data = io.BytesIO(file_bytes)
        
        # Upload the wheel file to the volume
        wheel_filename = os.path.basename(wheel_file_path)
        volume_file_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{wheel_filename}"
        
        logger.info(f"Uploading wheel file to: {volume_file_path}")
        
        w.files.upload(volume_file_path, binary_data, overwrite=True)
        
        logger.info(f"Successfully uploaded wheel file to: {volume_file_path}")
        
        # List files in the volume to verify (using correct method from SDK docs)
        try:
            directory_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"
            files = w.files.list_directory_contents(directory_path)
            logger.info("Files in volume:")
            for file in files:
                logger.info(f"  - {file.path}")
        except Exception as e:
            logger.warning(f"Could not list directory contents: {str(e)}")
        
        return volume_file_path
        
    except Exception as e:
        logger.error(f"Error uploading wheel file: {str(e)}")
        raise

def install_wheel_in_cluster(volume_path):
    """
    Example function showing how to install the wheel in a cluster.
    This would typically be done in a notebook or job.
    """
    logger.info("To install the wheel in a cluster, you can use:")
    logger.info(f"pip install {volume_path}")

def main():
    """Main function to handle command line arguments and execute upload."""
    parser = argparse.ArgumentParser(
        description="Upload a wheel file to a Unity Catalog volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_wheel_to_volume.py --catalog daveok --schema default --volume wheels --file /path/to/wheel.whl
  python upload_wheel_to_volume.py -c daveok -s default -v wheels -f /path/to/wheel.whl
        """
    )
    
    parser.add_argument(
        "--catalog", "-c",
        required=True,
        help="Unity Catalog catalog name"
    )
    
    parser.add_argument(
        "--schema", "-s", 
        required=True,
        help="Schema name within the catalog"
    )
    
    parser.add_argument(
        "--volume", "-v",
        required=True,
        help="Volume name within the schema"
    )
    
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the wheel file to upload"
    )
    
    args = parser.parse_args()
    
    try:
        # Get workspace URL for display
        w = WorkspaceClient()
        workspace_url = w.config.host
        
        volume_path = upload_wheel_to_volume(
            args.catalog, 
            args.schema, 
            args.volume, 
            args.file
        )
        print(f"\n✅ Wheel file uploaded successfully!")
        print(f"🌐 Workspace: {workspace_url}")
        print(f"📁 Location: {volume_path}")
        print(f"\nTo install in a cluster, use:")
        print(f"pip install {volume_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()