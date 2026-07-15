import AVFoundation
import SimulatorCameraClient
import SwiftUI

@Observable
final class CameraManager: NSObject {
    var isRunning = false
    var permissionDenied = false

    /// Simulator has no real AVCaptureSession frames to show on previewLayer,
    /// so the view falls back to rendering this instead. Unused on a device.
    var previewImage: UIImage?

    let previewLayer = AVCaptureVideoPreviewLayer()
    private let ciContext = CIContext()

    nonisolated(unsafe) private let session = AVCaptureSession()
    nonisolated(unsafe) private let videoOutput = AVCaptureVideoDataOutput()

    // Simulator has no camera hardware — SimulatorCameraOutput bypasses
    // AVCaptureSession entirely and streams frames from a companion Mac app.
    nonisolated(unsafe) private let simulatorOutput = SimulatorCameraOutput()
    nonisolated(unsafe) private var simulatorCaptureStarted = false
    nonisolated(unsafe) private var latestPixelBuffer: CVPixelBuffer?

    nonisolated(unsafe) private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    nonisolated(unsafe) private let outputQueue = DispatchQueue(label: "camera.output.queue", qos: .userInitiated)

    override init() {
        super.init()
        setupSession()
    }

    private func setupSession() {
        session.sessionPreset = .hd1920x1080
        previewLayer.session = session
        previewLayer.videoGravity = .resizeAspectFill

        // No-op on a real device. On Simulator, connects to the companion
        // app on the Mac (default 127.0.0.1:9876).
        SimulatorCamera.configure(host: "127.0.0.1", port: 9876)
    }

    nonisolated private func configureInputOutput() {
        #if targetEnvironment(simulator)
        simulatorOutput.setSampleBufferDelegate(self, queue: outputQueue)
        #else
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            print("CameraManager: failed to set up camera input")
            return
        }

        session.beginConfiguration()
        session.addInput(input)

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)

        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        session.commitConfiguration()

        // The back camera sensor is natively landscape. Without this, both the
        // live preview and the delivered sample buffers come out rotated 90°.
        for connection in [previewLayer.connection, videoOutput.connection(with: .video)] {
            guard let connection, connection.isVideoRotationAngleSupported(90) else { continue }
            connection.videoRotationAngle = 90
        }
        #endif
    }

    func startSession() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startCapture()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    self?.startCapture()
                } else {
                    DispatchQueue.main.async { self?.permissionDenied = true }
                }
            }
        default:
            DispatchQueue.main.async { self.permissionDenied = true }
        }
    }

    private func startCapture() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            #if targetEnvironment(simulator)
            guard !self.simulatorCaptureStarted else { return }
            self.simulatorCaptureStarted = true
            self.configureInputOutput()
            SimulatorCamera.start()
            #else
            guard !self.session.isRunning else { return }
            if self.session.inputs.isEmpty {
                self.configureInputOutput()
            }
            self.session.startRunning()
            #endif
            DispatchQueue.main.async { self.isRunning = true }
        }
    }

    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            #if targetEnvironment(simulator)
            guard self.simulatorCaptureStarted else { return }
            self.simulatorCaptureStarted = false
            SimulatorCamera.stop()
            #else
            guard self.session.isRunning else { return }
            self.session.stopRunning()
            #endif
            DispatchQueue.main.async { self.isRunning = false }
        }
    }

    /// Grabs the most recent live frame as a still photo. Returns nil if no frame has arrived yet.
    func capturePhoto() -> UIImage? {
        guard let pixelBuffer = latestPixelBuffer else { return nil }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        #if targetEnvironment(simulator)
        return UIImage(cgImage: cgImage, scale: 1, orientation: .right)
        #else
        // Already rotated upright by the connection's videoRotationAngle.
        return UIImage(cgImage: cgImage, scale: 1, orientation: .up)
        #endif
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        latestPixelBuffer = pixelBuffer

        #if targetEnvironment(simulator)
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        if let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) {
            let image = UIImage(cgImage: cgImage, scale: 1, orientation: .right)
            DispatchQueue.main.async { [weak self] in
                self?.previewImage = image
            }
        }
        #endif
    }
}
