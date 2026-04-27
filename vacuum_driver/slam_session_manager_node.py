import rclpy
from rclpy.node import Node
from slam_toolbox.srv import ClearQueue, Reset


class SlamSessionManagerNode(Node):
    def __init__(self):
        super().__init__('slam_session_manager_node')

        self.declare_parameter('reset_service', '/slam_toolbox/reset')
        self.declare_parameter('clear_queue_service', '/slam_toolbox/clear_queue')
        self.declare_parameter('startup_delay_sec', 2.0)
        self.declare_parameter('service_wait_timeout_sec', 20.0)
        self.declare_parameter('retry_period_sec', 0.5)
        self.declare_parameter('shutdown_after_reset', True)

        self.reset_service = str(self.get_parameter('reset_service').value)
        self.clear_queue_service = str(self.get_parameter('clear_queue_service').value)
        self.startup_delay_sec = max(
            0.0, float(self.get_parameter('startup_delay_sec').value)
        )
        self.service_wait_timeout_sec = max(
            1.0, float(self.get_parameter('service_wait_timeout_sec').value)
        )
        self.retry_period_sec = max(
            0.1, float(self.get_parameter('retry_period_sec').value)
        )
        self.shutdown_after_reset = bool(
            self.get_parameter('shutdown_after_reset').value
        )

        self.started_at = self.now_sec()
        self.last_wait_log_at = 0.0
        self.request_sent = False
        self.finished = False

        self.reset_client = self.create_client(Reset, self.reset_service)
        self.clear_queue_client = self.create_client(
            ClearQueue, self.clear_queue_service
        )
        self.timer = self.create_timer(self.retry_period_sec, self._tick)

    def now_sec(self):
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _tick(self):
        if self.finished or self.request_sent:
            return

        now = self.now_sec()
        elapsed = now - self.started_at
        if elapsed < self.startup_delay_sec:
            return

        if not self.reset_client.wait_for_service(timeout_sec=0.0):
            if (now - self.last_wait_log_at) > 2.0:
                self.get_logger().info(
                    f'Waiting for SLAM reset service: {self.reset_service}'
                )
                self.last_wait_log_at = now
            if elapsed > self.service_wait_timeout_sec:
                self.get_logger().warn(
                    'Timed out waiting for slam_toolbox reset service. '
                    'Mapping will continue without forced session reset.'
                )
                self._finish()
            return

        self.request_sent = True
        self.timer.cancel()
        self.get_logger().info('Starting fresh SLAM session reset.')

        if self.clear_queue_client.wait_for_service(timeout_sec=0.0):
            future = self.clear_queue_client.call_async(ClearQueue.Request())
            future.add_done_callback(self._on_clear_queue_done)
            return

        self._call_reset()

    def _on_clear_queue_done(self, future):
        try:
            response = future.result()
            if not response.status:
                self.get_logger().warn('slam_toolbox clear_queue returned false.')
        except Exception as exc:
            self.get_logger().warn(f'clear_queue service failed: {exc}')

        self._call_reset()

    def _call_reset(self):
        request = Reset.Request()
        request.pause_new_measurements = False
        future = self.reset_client.call_async(request)
        future.add_done_callback(self._on_reset_done)

    def _on_reset_done(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f'slam_toolbox reset failed: {exc}')
            self._finish()
            return

        if response.result == Reset.Response.RESULT_SUCCESS:
            self.get_logger().info('slam_toolbox reset complete. Fresh map session ready.')
        else:
            self.get_logger().error(
                f'slam_toolbox reset returned non-success result: {response.result}'
            )
        self._finish()

    def _finish(self):
        self.finished = True
        if self.shutdown_after_reset and rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = SlamSessionManagerNode()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        else:
            node.destroy_node()


if __name__ == '__main__':
    main()
