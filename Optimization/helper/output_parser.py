
def output_log_parser(log_file):
    """Extract wall time (in cycles) from AstraSim log file. Returns tuple (max_time_seconds, is_any_oom)."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for wall time in cycles from all systems
        import re
        matches_exec = re.findall(r'\[statistics\] \[info\] sys\[(\d+)\], Wall time: (\d+)', content)
        matches_oom = re.findall(r'\[workload\] \[info\] sys\[(\d+)\] is OOM: (\d+)', content)
        matches_peak_memory = re.findall(r'\[workload\] \[info\] sys\[(\d+)\] peak memory usage: (\d+)', content)

        slowest_npu = None
        is_any_oom = None
        peak_memory = None
        if matches_exec and matches_oom:
            # Extract all execution times
            exec_times = [int(cycles) for sys_id, cycles in matches_exec]
            
            # Return max execution time and 1 if any system is OOM
            slowest_npu = max(exec_times)
        
        if matches_exec and matches_oom:
            # Extract all OOM statuses
            oom_statuses = [int(oom_status) for sys_id, oom_status in matches_oom]
            
            is_any_oom = 1 if any(oom_statuses) else 0
            
        if matches_peak_memory:
            peak_memory = sum([float(cycles) for sys_id, cycles in matches_peak_memory])

        return slowest_npu, is_any_oom, peak_memory
    except Exception as e:
        print(f"Error parsing output: {e}")
        return None, None, None