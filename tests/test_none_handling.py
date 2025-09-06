"""
Test None value handling in time series data.

Simple tests to verify that None values are preserved correctly.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

try:
    import pandas as pd
except ImportError:
    pd = None

from openelectricity.models.timeseries import (
    TimeSeriesDataPoint,
    TimeSeriesResult,
    NetworkTimeSeries,
    TimeSeriesResponse,
    fix_none_values_in_data,
)


class TestNoneHandling:
    """Test None value preservation in time series data."""

    def test_fix_none_values_preserves_none(self):
        """Test that fix_none_values_in_data preserves None values."""
        # Test data with None values
        test_data = {
            "data": [
                (datetime.now(), 100.0),
                (datetime.now(), None),  # This should be preserved
                (datetime.now(), 200.0),
            ]
        }
        
        result = fix_none_values_in_data(test_data)
        
        # Check that None is preserved
        assert result["data"][1][1] is None
        assert result["data"][0][1] == 100.0
        assert result["data"][2][1] == 200.0

    def test_timeseries_datapoint_with_none(self):
        """Test that TimeSeriesDataPoint can handle None values."""
        timestamp = datetime.now()
        
        # Should work with None value
        point = TimeSeriesDataPoint((timestamp, None))
        assert point.timestamp == timestamp
        assert point.value is None

    def test_timeseries_result_with_none_data(self):
        """Test that TimeSeriesResult can handle None values in data."""
        timestamp = datetime.now()
        
        result_data = {
            "name": "test_metric",
            "date_start": timestamp,
            "date_end": timestamp,
            "columns": {"network_region": "NSW1"},
            "data": [
                (timestamp, 100.0),
                (timestamp, None),  # None value should be preserved
                (timestamp, 200.0),
            ]
        }
        
        result = TimeSeriesResult.model_validate(result_data)
        
        # Check that None is preserved
        assert result.data[1].value is None
        assert result.data[0].value == 100.0
        assert result.data[2].value == 200.0

    def test_network_timeseries_with_none_data(self):
        """Test that NetworkTimeSeries can handle None values."""
        timestamp = datetime.now()
        
        timeseries_data = {
            "network_code": "NEM",
            "metric": "power",
            "unit": "MW",
            "interval": "5m",
            "network_timezone_offset": "+10:00",
            "results": [
                {
                    "name": "power.coal_black",
                    "date_start": timestamp,
                    "date_end": timestamp,
                    "columns": {"network_region": "NSW1"},
                    "data": [
                        (timestamp, 1000.0),
                        (timestamp, None),  # None value should be preserved
                        (timestamp, 950.0),
                    ]
                }
            ]
        }
        
        timeseries = NetworkTimeSeries.model_validate(timeseries_data)
        
        # Check that None is preserved
        assert timeseries.results[0].data[1].value is None
        assert timeseries.results[0].data[0].value == 1000.0
        assert timeseries.results[0].data[2].value == 950.0

    def test_timeseries_response_with_none_data(self):
        """Test that TimeSeriesResponse can handle None values."""
        timestamp = datetime.now()
        
        response_data = {
            "version": "1.0",
            "created_at": timestamp,
            "data": [
                {
                    "network_code": "NEM",
                    "metric": "power",
                    "unit": "MW",
                    "interval": "5m",
                    "network_timezone_offset": "+10:00",
                    "results": [
                        {
                            "name": "power.coal_black",
                            "date_start": timestamp,
                            "date_end": timestamp,
                            "columns": {"network_region": "NSW1"},
                            "data": [
                                (timestamp, 1000.0),
                                (timestamp, None),  # None value should be preserved
                                (timestamp, 950.0),
                            ]
                        }
                    ]
                }
            ]
        }
        
        response = TimeSeriesResponse.model_validate(response_data)
        
        # Check that None is preserved
        assert response.data[0].results[0].data[1].value is None
        assert response.data[0].results[0].data[0].value == 1000.0
        assert response.data[0].results[0].data[2].value == 950.0

    def test_none_values_in_to_records(self):
        """Test that None values are preserved when converting to records."""
        from datetime import timedelta
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(minutes=5)
        
        response_data = {
            "version": "1.0",
            "created_at": timestamp1,
            "data": [
                {
                    "network_code": "NEM",
                    "metric": "power",
                    "unit": "MW",
                    "interval": "5m",
                    "network_timezone_offset": "+10:00",
                    "results": [
                        {
                            "name": "power.coal_black",
                            "date_start": timestamp1,
                            "date_end": timestamp2,
                            "columns": {"network_region": "NSW1"},
                            "data": [
                                (timestamp1, 1000.0),
                                (timestamp2, None),  # None value should be preserved
                            ]
                        }
                    ]
                }
            ]
        }
        
        response = TimeSeriesResponse.model_validate(response_data)
        records = response.to_records()
        
        # Check that None is preserved in records
        assert len(records) == 2
        # Find the record with None value
        none_record = next(r for r in records if r["power"] is None)
        value_record = next(r for r in records if r["power"] == 1000.0)
        assert none_record["power"] is None
        assert value_record["power"] == 1000.0

    def test_none_values_in_dataframe_conversion(self):
        """Test that None values are preserved when converting to DataFrame."""
        from datetime import timedelta
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(minutes=5)
        
        response_data = {
            "version": "1.0",
            "created_at": timestamp1,
            "data": [
                {
                    "network_code": "NEM",
                    "metric": "power",
                    "unit": "MW",
                    "interval": "5m",
                    "network_timezone_offset": "+10:00",
                    "results": [
                        {
                            "name": "power.coal_black",
                            "date_start": timestamp1,
                            "date_end": timestamp2,
                            "columns": {"network_region": "NSW1"},
                            "data": [
                                (timestamp1, 1000.0),
                                (timestamp2, None),  # None value should be preserved
                            ]
                        }
                    ]
                }
            ]
        }
        
        response = TimeSeriesResponse.model_validate(response_data)
        
        # Test pandas conversion
        try:
            df = response.to_pandas()
            assert len(df) == 2
            # Find the row with None value
            none_row = df[df["power"].isna()]
            value_row = df[df["power"] == 1000.0]
            assert len(none_row) == 1
            assert len(value_row) == 1
            assert pd.isna(none_row["power"].iloc[0])  # None becomes NaN in pandas
            assert value_row["power"].iloc[0] == 1000.0
        except ImportError:
            pytest.skip("Pandas not available")

    def test_validation_error_handling_with_none(self):
        """Test that validation errors are handled gracefully with None values."""
        # This should not raise an exception even with None values
        timestamp = datetime.now()
        
        response_data = {
            "version": "1.0",
            "created_at": timestamp,
            "data": [
                {
                    "network_code": "NEM",
                    "metric": "power",
                    "unit": "MW",
                    "interval": "5m",
                    "network_timezone_offset": "+10:00",
                    "results": [
                        {
                            "name": "power.coal_black",
                            "date_start": timestamp,
                            "date_end": timestamp,
                            "columns": {"network_region": "NSW1"},
                            "data": [
                                (timestamp, None),  # None value should not cause validation error
                            ]
                        }
                    ]
                }
            ]
        }
        
        # This should not raise an exception
        response = TimeSeriesResponse.model_validate(response_data)
        assert response.data[0].results[0].data[0].value is None
