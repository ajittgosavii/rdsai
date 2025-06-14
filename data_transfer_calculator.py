# data_transfer_calculator.py
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class TransferMethod(Enum):
    DATASYNC_DX = "datasync_dx"
    DATASYNC_INTERNET = "datasync_internet"
    SNOWBALL = "snowball"
    SNOWBALL_EDGE = "snowball_edge"
    SNOWMOBILE = "snowmobile"

@dataclass
class TransferResult:
    transfer_time_hours: float
    transfer_time_days: float
    total_cost: float
    cost_breakdown: Dict[str, float]
    bandwidth_utilization: float
    recommended_method: str
    estimated_downtime_hours: float

class DataTransferCalculator:
    def __init__(self):
        # AWS DataSync pricing (per GB transferred)
        self.datasync_pricing = {
            'us-east-1': 0.0125,  # $0.0125 per GB
            'us-west-1': 0.0125,
            'us-west-2': 0.0125,
            'eu-west-1': 0.0125,
            'eu-central-1': 0.0125,
            'ap-southeast-1': 0.0125,
        }
        
        # Direct Connect pricing (per GB transferred out)
        self.dx_transfer_pricing = {
            'us-east-1': 0.02,  # $0.02 per GB
            'us-west-1': 0.02,
            'us-west-2': 0.02,
            'eu-west-1': 0.02,
            'eu-central-1': 0.02,
            'ap-southeast-1': 0.02,
        }
        
        # Direct Connect port pricing (per hour)
        self.dx_port_pricing = {
            '1Gbps': 0.30,      # $0.30 per hour
            '10Gbps': 2.25,     # $2.25 per hour
            '100Gbps': 22.50,   # $22.50 per hour
        }
        
        # AWS Snow family pricing
        self.snow_pricing = {
            'snowball': 200,        # $200 per job
            'snowball_edge': 300,   # $300 per job
            'snowmobile': 0.005,    # $0.005 per GB per month
        }
        
        # Transfer efficiency factors
        self.efficiency_factors = {
            'datasync_dx': 0.85,           # 85% efficiency for DataSync over DX
            'datasync_internet': 0.70,     # 70% efficiency for DataSync over internet
            'snowball': 1.0,               # Physical transfer
            'dx_raw': 0.90,                # 90% efficiency for raw DX transfer
        }
        
        # Compression ratios by data type
        self.compression_ratios = {
            'database': 0.7,    # 30% compression for database files
            'logs': 0.5,        # 50% compression for log files
            'mixed': 0.8,       # 20% compression for mixed data
            'none': 1.0,        # No compression
        }

    def calculate_transfer_time(
        self, 
        data_size_gb: float, 
        bandwidth_gbps: float, 
        method: TransferMethod = TransferMethod.DATASYNC_DX,
        compression_ratio: float = 0.8
    ) -> Tuple[float, float]:
        """
        Calculate transfer time in hours and days
        
        Args:
            data_size_gb: Size of data in GB
            bandwidth_gbps: Available bandwidth in Gbps
            method: Transfer method
            compression_ratio: Data compression ratio (0.8 = 20% compression)
        
        Returns:
            Tuple of (hours, days)
        """
        # Apply compression
        effective_data_size_gb = data_size_gb * compression_ratio
        
        # Apply efficiency factor
        efficiency = self.efficiency_factors.get(method.value, 0.85)
        effective_bandwidth_gbps = bandwidth_gbps * efficiency
        
        # Convert GB to Gb (bits)
        data_size_gb_bits = effective_data_size_gb * 8
        
        # Calculate transfer time in hours
        transfer_time_hours = data_size_gb_bits / (effective_bandwidth_gbps * 3600)  # 3600 seconds per hour
        transfer_time_days = transfer_time_hours / 24
        
        return transfer_time_hours, transfer_time_days

    def calculate_datasync_cost(
        self, 
        data_size_gb: float, 
        region: str = 'us-east-1',
        compression_ratio: float = 0.8
    ) -> float:
        """Calculate AWS DataSync cost"""
        effective_data_size_gb = data_size_gb * compression_ratio
        cost_per_gb = self.datasync_pricing.get(region, 0.0125)
        return effective_data_size_gb * cost_per_gb

    def calculate_dx_cost(
        self, 
        data_size_gb: float, 
        transfer_time_hours: float,
        region: str = 'us-east-1',
        port_speed: str = '10Gbps',
        compression_ratio: float = 0.8
    ) -> Dict[str, float]:
        """Calculate Direct Connect costs"""
        effective_data_size_gb = data_size_gb * compression_ratio
        
        # Transfer cost (data out)
        transfer_cost_per_gb = self.dx_transfer_pricing.get(region, 0.02)
        transfer_cost = effective_data_size_gb * transfer_cost_per_gb
        
        # Port cost (time-based)
        port_cost_per_hour = self.dx_port_pricing.get(port_speed, 2.25)
        port_cost = transfer_time_hours * port_cost_per_hour
        
        return {
            'transfer_cost': transfer_cost,
            'port_cost': port_cost,
            'total_dx_cost': transfer_cost + port_cost
        }

    def calculate_comprehensive_transfer_analysis(
        self,
        data_size_gb: float,
        region: str = 'us-east-1',
        dx_bandwidth_gbps: float = 10.0,
        internet_bandwidth_mbps: float = 100.0,
        compression_type: str = 'database'
    ) -> Dict[str, TransferResult]:
        """
        Perform comprehensive transfer analysis for all methods
        """
        results = {}
        compression_ratio = self.compression_ratios.get(compression_type, 0.8)
        
        # DataSync over Direct Connect (10 Gbps)
        dx_hours, dx_days = self.calculate_transfer_time(
            data_size_gb, dx_bandwidth_gbps, TransferMethod.DATASYNC_DX, compression_ratio
        )
        
        datasync_cost = self.calculate_datasync_cost(data_size_gb, region, compression_ratio)
        dx_costs = self.calculate_dx_cost(data_size_gb, dx_hours, region, '10Gbps', compression_ratio)
        
        total_dx_cost = datasync_cost + dx_costs['total_dx_cost']
        
        results['datasync_dx'] = TransferResult(
            transfer_time_hours=dx_hours,
            transfer_time_days=dx_days,
            total_cost=total_dx_cost,
            cost_breakdown={
                'datasync_cost': datasync_cost,
                'dx_transfer_cost': dx_costs['transfer_cost'],
                'dx_port_cost': dx_costs['port_cost']
            },
            bandwidth_utilization=85.0,  # 85% efficiency
            recommended_method="DataSync over 10Gbps Direct Connect",
            estimated_downtime_hours=min(dx_hours * 0.1, 4)  # 10% of transfer time or 4 hours max
        )
        
        # DataSync over Internet
        internet_bandwidth_gbps = internet_bandwidth_mbps / 1000  # Convert Mbps to Gbps
        internet_hours, internet_days = self.calculate_transfer_time(
            data_size_gb, internet_bandwidth_gbps, TransferMethod.DATASYNC_INTERNET, compression_ratio
        )
        
        # Internet transfer cost (typically free inbound, DataSync charges apply)
        internet_total_cost = datasync_cost
        
        results['datasync_internet'] = TransferResult(
            transfer_time_hours=internet_hours,
            transfer_time_days=internet_days,
            total_cost=internet_total_cost,
            cost_breakdown={
                'datasync_cost': datasync_cost,
                'internet_cost': 0.0
            },
            bandwidth_utilization=70.0,  # 70% efficiency
            recommended_method=f"DataSync over Internet ({internet_bandwidth_mbps} Mbps)",
            estimated_downtime_hours=min(internet_hours * 0.1, 8)  # 10% of transfer time or 8 hours max
        )
        
        # AWS Snowball (for large datasets)
        if data_size_gb > 10000:  # 10TB threshold
            snowball_days = 7  # Typical shipping time + transfer
            snowball_hours = snowball_days * 24
            snowball_cost = self.snow_pricing['snowball']
            
            results['snowball'] = TransferResult(
                transfer_time_hours=snowball_hours,
                transfer_time_days=snowball_days,
                total_cost=snowball_cost,
                cost_breakdown={
                    'snowball_service_cost': snowball_cost,
                    'shipping_cost': 0.0  # Included in service cost
                },
                bandwidth_utilization=100.0,  # Physical transfer
                recommended_method="AWS Snowball (Physical Transfer)",
                estimated_downtime_hours=24  # 1 day downtime for cutover
            )
        
        return results

    def get_recommended_method(
        self, 
        data_size_gb: float, 
        time_sensitivity: str = 'medium',
        cost_sensitivity: str = 'medium'
    ) -> str:
        """
        Recommend best transfer method based on data size and requirements
        
        Args:
            data_size_gb: Size of data in GB
            time_sensitivity: 'low', 'medium', 'high'
            cost_sensitivity: 'low', 'medium', 'high'
        """
        
        if data_size_gb < 100:  # Less than 100GB
            return "DataSync over Internet"
        elif data_size_gb < 1000:  # 100GB - 1TB
            if time_sensitivity == 'high':
                return "DataSync over Direct Connect"
            else:
                return "DataSync over Internet"
        elif data_size_gb < 10000:  # 1TB - 10TB
            if cost_sensitivity == 'high':
                return "AWS Snowball"
            else:
                return "DataSync over Direct Connect"
        else:  # More than 10TB
            if time_sensitivity == 'high' and cost_sensitivity == 'low':
                return "Multiple DataSync over Direct Connect (Parallel)"
            else:
                return "AWS Snowball or Snowball Edge"

    def calculate_parallel_transfer_optimization(
        self,
        data_size_gb: float,
        available_bandwidth_gbps: float = 10.0,
        max_parallel_tasks: int = 10
    ) -> Dict[str, any]:
        """
        Calculate optimization for parallel transfers
        """
        # Optimal number of parallel DataSync tasks
        optimal_tasks = min(max_parallel_tasks, math.ceil(data_size_gb / 1000))  # 1 task per TB
        
        # Bandwidth per task
        bandwidth_per_task = available_bandwidth_gbps / optimal_tasks
        
        # Calculate transfer time with parallel tasks
        parallel_hours, parallel_days = self.calculate_transfer_time(
            data_size_gb / optimal_tasks, 
            bandwidth_per_task, 
            TransferMethod.DATASYNC_DX
        )
        
        return {
            'optimal_parallel_tasks': optimal_tasks,
            'bandwidth_per_task_gbps': bandwidth_per_task,
            'transfer_time_hours': parallel_hours,
            'transfer_time_days': parallel_days,
            'efficiency_improvement': f"{((data_size_gb/10) / parallel_hours - 1) * 100:.1f}%"
        }

# Example usage and testing
if __name__ == "__main__":
    calculator = DataTransferCalculator()
    
    # Test with 500GB database
    test_size_gb = 500
    results = calculator.calculate_comprehensive_transfer_analysis(
        data_size_gb=test_size_gb,
        region='us-east-1',
        dx_bandwidth_gbps=10.0,
        internet_bandwidth_mbps=100.0,
        compression_type='database'
    )
    
    print(f"Transfer Analysis for {test_size_gb}GB Database:")
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Time: {result.transfer_time_hours:.1f} hours ({result.transfer_time_days:.1f} days)")
        print(f"  Cost: ${result.total_cost:.2f}")
        print(f"  Method: {result.recommended_method}")
        print(f"  Downtime: {result.estimated_downtime_hours:.1f} hours")