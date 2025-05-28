import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class RecursivePIDController:
    def __init__(self, Kp, Ki, Kd):
        """
        递归式PID控制器
        :param Kp: 比例增益
        :param Ki: 积分增益
        :param Kd: 微分增益
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.reset_state()

    def reset_state(self):
        """重置控制器状态"""
        self.last_error = 0
        self.integral = 0
        self._current_step = 0

    def compute(self, setpoint, measured_value, dt, max_recursion=20, _recursion_depth=0):
        """
        递归式PID计算
        :param setpoint: 设定值
        :param measured_value: 测量值
        :param dt: 时间步长
        :param max_recursion: 最大递归深度
        :param _recursion_depth: 当前递归深度(内部使用)
        :return: (控制输出, 测量值)
        """
        if _recursion_depth >= max_recursion:
            error = setpoint - measured_value
            derivative = (error - self.last_error) / dt if dt > 0 else 0
            return (self.Kp * error + self.Ki * self.integral + self.Kd * derivative, measured_value)

        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        self._current_step += 1

        return self.compute(setpoint, measured_value, dt, max_recursion, _recursion_depth + 1)

    def recursive_simulation(self, setpoint_func, system_func, dt, total_time, current_t=0, results=None):
        """
        递归式系统仿真
        :param setpoint_func: 设定值函数 setpoint(t)
        :param system_func: 系统函数 y_next = f(y, u, dt)
        :param dt: 时间步长
        :param total_time: 总仿真时间
        :param current_t: 当前时间(内部使用)
        :param results: 结果存储(内部使用)
        :return: 仿真结果字典
        """
        if results is None:
            results = {
                'time': [],
                'output': [],
                'setpoint': [],
                'control': [],
                'error': [],
                'integral': []
            }

        if current_t > total_time + 1e-6:  # 处理浮点数精度问题
            return results

        # 获取当前设定值
        current_setpoint = setpoint_func(current_t)
        
        # 获取当前测量值(上一步的输出)
        current_measured = results['output'][-1] if results['output'] else 0
        
        # 计算控制量
        u, _ = self.compute(current_setpoint, current_measured, dt)
        
        # 系统响应
        new_y = system_func(current_measured, u, dt)
        
        # 记录结果
        results['time'].append(current_t)
        results['output'].append(new_y)
        results['setpoint'].append(current_setpoint)
        results['control'].append(u)
        results['error'].append(current_setpoint - current_measured)
        results['integral'].append(self.integral)
        
        # 递归进入下一时间步
        return self.recursive_simulation(
            setpoint_func, system_func, dt, 
            total_time, current_t + dt, results
        )

def plot_results(results, pid_params):
    """绘制仿真结果"""
    plt.figure(figsize=(12, 20))
    gs = GridSpec(3, 2, figure=plt.gcf())
    
    # 系统响应曲线
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(results['time'], results['output'], label='System Output')
    ax1.plot(results['time'], results['setpoint'], 'r--', label='Setpoint')
    ax1.set_title('System Response (Kp={:.2f}, Ki={:.2f}, Kd={:.2f})'.format(
        pid_params['Kp'], pid_params['Ki'], pid_params['Kd']))
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # 控制信号
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(results['time'], results['control'], 'g-', label='Control Signal')
    ax2.set_title('Control Signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Value')
    ax2.legend()
    ax2.grid(True)
    
    # 误差曲线
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(results['time'], results['error'], 'm-', label='Error')
    ax3.set_title('Error')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error')
    ax3.legend()
    ax3.grid(True)
    
    # 积分项
    ax4 = plt.subplot(gs[2, :])
    ax4.plot(results['time'], results['integral'], 'b-', label='Integral Term')
    ax4.set_title('Integral Term Accumulation')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Integral Value')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def system_model(y, u, dt, tau=0.5):
    """一阶系统模型"""
    return y + (u - y)/tau * dt

def setpoint_function(t):
    """时变设定值函数"""
    if t < 2.0:
        return 1.0
    elif t < 4.0:
        return 0.5
    else:
        return 1.5

if __name__ == "__main__":
    # PID参数
    pid_params = {
        'Kp': 1.5,
        'Ki': 0.2,
        'Kd': 0.1
    }
    
    # 创建递归PID控制器
    pid = RecursivePIDController(**pid_params)
    
    # 运行递归仿真
    results = pid.recursive_simulation(
        setpoint_func=setpoint_function,
        system_func=system_model,
        dt=0.05,
        total_time=20.0
    )
    
    # 绘制结果
    plot_results(results, pid_params)