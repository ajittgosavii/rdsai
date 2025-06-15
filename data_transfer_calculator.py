import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

class TransferMethod(Enum):
    """Enum for different data transfer methods."""
    DATASYNC_DX = "AWS DataSync (Direct Connect)"
    DATASYNC_INTERNET = "AWS DataSync (Internet)"
    SNOWBALL = "AWS Snowball"
    SNOWBALL_EDGE = "AWS Snowball Edge"
    DIRECT_UPLOAD = "Direct Internet Upload" # For smaller transfers

@dataclass
class TransferMethodResult:
    """Represents the result of a data transfer calculation for a specific method."""
    recommended_method: str
    transfer_time_hours: float
    transfer_time_days: float
    total_cost: float
    bandwidth_utilization: float
    estimated_downtime_hours: float
    cost_breakdown: Dict[str, float]


class DataTransferCalculator:
    """
    Calculates and compares data transfer options for migrating data to AWS.
    This class provides estimations for transfer time, cost, and other metrics
    for various AWS data migration services.
    """

    def __init__(self):
        # Base pricing (simplified for demonstration)
        # In a real application, this would fetch from AWS Pricing API or a database
        self.pricing = {
            "us-east-1": {
                "datasync_internet_gb": 0.0125,  # per GB after free tier
                "datasync_dx_gb": 0.0125,        # per GB after free tier (same as internet for DataSync)
                "s3_storage_gb_month": 0.023,    # Standard S3, first 50 TB
                "dx_port_hour": 0.30,            # 1Gbps Direct Connect port hour
                "dx_data_egress_gb": 0.02,       # Data transfer out via DX
                "internet_data_egress_gb": 0.09, # Data transfer out via internet (higher)
                "snowball_appliance_fee": 200,   # Snowball (80 TB) appliance fee
                "snowball_edge_appliance_fee": 300, # Snowball Edge (100 TB) appliance fee
                "snowball_shipping_fee": 50,     # Placeholder shipping fee
                "snowball_edge_shipping_fee": 75, # Placeholder shipping fee
            }
        }
        self.region = "us-east-1" # Default region

    def set_region(self, region: str):
        """Sets the AWS region for pricing lookups."""
        if region in self.pricing:
            self.region = region
        else:
            print(f"Warning: Pricing data not available for region {region}. Using default {self.region}.")

    def _get_pricing(self, service_key: str) -> float:
        """Retrieves pricing for a given service key in the current region."""
        return self.pricing.get(self.region, {}).get(service_key, 0.0)

    def calculate_datasync_transfer(self, data_size_gb: float, bandwidth_mbps: float, 
                                     is_direct_connect: bool = False, compression_ratio: float = 1.0) -> TransferMethodResult:
        """
        Calculates transfer time and cost for AWS DataSync.
        
        Args:
            data_size_gb: Total data size in GB.
            bandwidth_mbps: Available network bandwidth in Mbps.
            is_direct_connect: True if using Direct Connect, False for Internet.
            compression_ratio: How much data is compressed (e.g., 0.5 for 50% reduction).
        
        Returns:
            A TransferMethodResult object.
        """
        if bandwidth_mbps <= 0:
            return TransferMethodResult(
                recommended_method=TransferMethod.DATASYNC_DX.value if is_direct_connect else TransferMethod.DATASYNC_INTERNET.value,
                transfer_time_hours=float('inf'),
                transfer_time_days=float('inf'),
                total_cost=float('inf'),
                bandwidth_utilization=0.0,
                estimated_downtime_hours=0.0,
                cost_breakdown={"error": "Bandwidth must be positive"}
            )

        # Convert data size to Mbits for calculation
        # 1 GB = 8192 Mbits
        effective_data_mbits = (data_size_gb * 8192) * compression_ratio
        
        # Calculate time in hours
        # Time = Data (Mbits) / Bandwidth (Mbits/sec) / 3600 (sec/hour)
        transfer_time_seconds = effective_data_mbits / bandwidth_mbps
        transfer_time_hours = transfer_time_seconds / 3600
        transfer_time_days = transfer_time_hours / 24

        # Calculate DataSync cost (simplified - DataSync charges per GB transferred)
        datasync_cost_per_gb = self._get_pricing("datasync_internet_gb") # Same pricing whether DX or Internet
        datasync_task_cost = data_size_gb * datasync_cost_per_gb

        # Add potential egress costs if data is coming from another cloud or on-prem that charges egress
        # For simplicity, assuming no egress from source for this tool's scope unless specified.
        # This example primarily covers ingress to AWS S3 via DataSync.
        
        total_cost = datasync_task_cost
        cost_breakdown = {"datasync_task": datasync_task_cost}
        
        # Bandwidth Utilization (simplistic: assume 80% effective utilization)
        bandwidth_utilization = min(100.0, (effective_data_mbits / (transfer_time_seconds * bandwidth_mbps)) * 100)

        # Estimated Downtime: DataSync typically has minimal downtime, usually during cutover.
        # This is a very rough estimate.
        estimated_downtime_hours = max(0.1, transfer_time_hours * 0.01) # 1% of transfer time, minimum 0.1 hours

        return TransferMethodResult(
            recommended_method=TransferMethod.DATASYNC_DX.value if is_direct_connect else TransferMethod.DATASYNC_INTERNET.value,
            transfer_time_hours=transfer_time_hours,
            transfer_time_days=transfer_time_days,
            total_cost=total_cost,
            bandwidth_utilization=bandwidth_utilization,
            estimated_downtime_hours=estimated_downtime_hours,
            cost_breakdown=cost_breakdown
        )

    def calculate_snowball_transfer(self, data_size_gb: float, snowball_type: str = "snowball") -> TransferMethodResult:
        """
        Calculates transfer time and cost for AWS Snowball/Snowball Edge.
        
        Args:
            data_size_gb: Total data size in GB.
            snowball_type: "snowball" (80 TB) or "snowball_edge" (100 TB).
        
        Returns:
            A TransferMethodResult object.
        """
        appliance_fee = self._get_pricing(f"{snowball_type}_appliance_fee")
        shipping_fee = self._get_pricing(f"{snowball_type}_shipping_fee")
        
        # Snowball capacities
        capacities = {
            "snowball": 80 * 1024,      # 80 TB to GB
            "snowball_edge": 100 * 1024 # 100 TB to GB
        }
        
        device_capacity_gb = capacities.get(snowball_type, 80 * 1024)
        
        # Number of devices needed
        num_devices = math.ceil(data_size_gb / device_capacity_gb)
        
        # Total cost
        total_cost = (num_devices * appliance_fee) + (num_devices * shipping_fee)
        cost_breakdown = {
            f"{snowball_type}_appliance_fees": num_devices * appliance_fee,
            f"{snowball_type}_shipping_fees": num_devices * shipping_fee
        }

        # Transfer time for Snowball (rough estimate for shipping + processing)
        # Typically 5-7 business days per direction, plus processing at AWS.
        # Let's estimate 2-3 weeks total (14-21 days) per device/transfer.
        transfer_time_days = num_devices * 18 # Avg 18 days per device cycle (shipping + ingest)
        transfer_time_hours = transfer_time_days * 24

        # Downtime for Snowball is usually minimal as it's an offline transfer,
        # but there might be a brief cutover.
        estimated_downtime_hours = 0.5 # A small fixed downtime for final cutover

        return TransferMethodResult(
            recommended_method=TransferMethod.SNOWBALL.value if snowball_type == "snowball" else TransferMethod.SNOWBALL_EDGE.value,
            transfer_time_hours=transfer_time_hours,
            transfer_time_days=transfer_time_days,
            total_cost=total_cost,
            bandwidth_utilization=100.0, # N/A for offline, but assume full utilization during ingest
            estimated_downtime_hours=estimated_downtime_hours,
            cost_breakdown=cost_breakdown
        )

    def calculate_direct_upload(self, data_size_gb: float, internet_bandwidth_mbps: float) -> TransferMethodResult:
        """
        Calculates transfer time and cost for direct internet upload.
        Suitable for smaller datasets or continuous replication where DataSync is overkill.
        
        Args:
            data_size_gb: Total data size in GB.
            internet_bandwidth_mbps: Available internet bandwidth in Mbps.
            
        Returns:
            A TransferMethodResult object.
        """
        if internet_bandwidth_mbps <= 0:
            return TransferMethodResult(
                recommended_method=TransferMethod.DIRECT_UPLOAD.value,
                transfer_time_hours=float('inf'),
                transfer_time_days=float('inf'),
                total_cost=float('inf'),
                bandwidth_utilization=0.0,
                estimated_downtime_hours=0.0,
                cost_breakdown={"error": "Bandwidth must be positive"}
            )

        # Convert data size to Mbits for calculation
        effective_data_mbits = data_size_gb * 8192
        
        # Calculate time in hours
        transfer_time_seconds = effective_data_mbits / internet_bandwidth_mbps
        transfer_time_hours = transfer_time_seconds / 3600
        transfer_time_days = transfer_time_hours / 24

        # Cost: Primarily S3 ingress (free) + potential egress if mirroring or transferring out.
        # For simplicity, focusing on cost to get data INTO S3, which is generally free.
        # But if there were any source egress costs, they'd be here.
        # We'll use a nominal fee or S3 storage cost if data is transient.
        # AWS S3 standard storage is ~ $0.023 per GB per month.
        # This is simplified: direct upload cost can be minimal unless special services are used.
        # Let's add a small nominal processing fee per GB for direct upload simulation.
        nominal_upload_processing_cost_per_gb = 0.005 # $0.005 per GB

        total_cost = data_size_gb * nominal_upload_processing_cost_per_gb
        cost_breakdown = {"upload_processing_fee": total_cost}
        
        # Bandwidth Utilization (simplistic: assume 70% effective utilization due to overhead)
        bandwidth_utilization = min(100.0, (effective_data_mbits / (transfer_time_seconds * internet_bandwidth_mbps)) * 0.7 * 100)

        # Downtime for direct upload can vary greatly depending on application.
        # For a full database dump/restore, it could be significant. For continuous sync, minimal.
        estimated_downtime_hours = max(0.2, transfer_time_hours * 0.1) # 10% of transfer time, min 0.2 hours

        return TransferMethodResult(
            recommended_method=TransferMethod.DIRECT_UPLOAD.value,
            transfer_time_hours=transfer_time_hours,
            transfer_time_days=transfer_time_days,
            total_cost=total_cost,
            bandwidth_utilization=bandwidth_utilization,
            estimated_downtime_hours=estimated_downtime_hours,
            cost_breakdown=cost_breakdown
        )

    def calculate_comprehensive_transfer_analysis(self, data_size_gb: float, region: str,
                                                  dx_bandwidth_gbps: float, internet_bandwidth_mbps: float,
                                                  compression_type: str = "database") -> Dict[str, TransferMethodResult]:
        """
        Performs a comprehensive analysis of various data transfer methods.
        
        Args:
            data_size_gb: Total data size in GB.
            region: AWS region for pricing.
            dx_bandwidth_gbps: Direct Connect bandwidth in Gbps.
            internet_bandwidth_mbps: Internet bandwidth in Mbps.
            compression_type: Type of data for estimating compression.
                              "database", "logs", "mixed", "none".
        
        Returns:
            A dictionary where keys are method identifiers and values are TransferMethodResult objects.
        """
        self.set_region(region)

        # Estimate compression ratio based on data type
        # These are rough estimates and can be highly variable
        compression_ratios = {
            "database": 0.6,  # 40% compression
            "logs": 0.3,      # 70% compression (logs are very compressible)
            "mixed": 0.7,     # 30% compression
            "none": 1.0       # No compression
        }
        effective_compression_ratio = compression_ratios.get(compression_type.lower(), 1.0)

        results = {}

        # Convert DX bandwidth from Gbps to Mbps for consistency
        dx_bandwidth_mbps = dx_bandwidth_gbps * 1024

        # 1. AWS DataSync over Direct Connect
        results['datasync_dx'] = self.calculate_datasync_transfer(
            data_size_gb=data_size_gb,
            bandwidth_mbps=dx_bandwidth_mbps,
            is_direct_connect=True,
            compression_ratio=effective_compression_ratio
        )
        results['datasync_dx'].recommended_method = "AWS DataSync (Direct Connect)"

        # 2. AWS DataSync over Internet
        results['datasync_internet'] = self.calculate_datasync_transfer(
            data_size_gb=data_size_gb,
            bandwidth_mbps=internet_bandwidth_mbps,
            is_direct_connect=False,
            compression_ratio=effective_compression_ratio
        )
        results['datasync_internet'].recommended_method = "AWS DataSync (Internet)"

        # 3. AWS Snowball (for large transfers > 10 TB)
        if data_size_gb >= 10 * 1024: # >= 10 TB
            results['snowball'] = self.calculate_snowball_transfer(data_size_gb, "snowball")
            results['snowball'].recommended_method = "AWS Snowball (80 TB)"

        # 4. AWS Snowball Edge (for very large transfers or edge compute needs)
        if data_size_gb >= 50 * 1024: # >= 50 TB
            results['snowball_edge'] = self.calculate_snowball_transfer(data_size_gb, "snowball_edge")
            results['snowball_edge'].recommended_method = "AWS Snowball Edge (100 TB)"
        
        # 5. Direct Internet Upload (for smaller data sizes or continuous sync not needing DataSync)
        # This is a fallback/option for situations where DataSync might be overkill.
        if data_size_gb < 5000: # < 5 TB
             results['direct_upload'] = self.calculate_direct_upload(data_size_gb, internet_bandwidth_mbps)
             results['direct_upload'].recommended_method = "Direct Internet Upload"


        return results

