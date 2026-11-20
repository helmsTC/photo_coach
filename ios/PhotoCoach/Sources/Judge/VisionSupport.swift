import CoreML
import Vision

extension VNRequest {
    /// The iOS Simulator cannot build Vision's GPU / Neural Engine contexts ("Failed to create espresso
    /// context"), so there every request is pinned to the CPU.  On a device Vision picks its own hardware.
    @discardableResult
    func cpuOnSimulator() -> Self {
        #if targetEnvironment(simulator)
        if let stages = try? supportedComputeStageDevices {
            for (stage, devices) in stages {
                if let cpu = devices.first(where: { if case .cpu = $0 { return true } else { return false } }) {
                    setComputeDevice(cpu, for: stage)
                }
            }
        }
        #endif
        return self
    }
}
