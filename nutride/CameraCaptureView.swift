import AVFoundation
import SwiftUI

struct CameraCaptureView: View {
    @State private var cameraManager = CameraManager()
    var onCapture: (UIImage) -> Void
    var onCancel: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            #if targetEnvironment(simulator)
            if let previewImage = cameraManager.previewImage {
                Image(uiImage: previewImage)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
            } else {
                ProgressView("Waiting for Simulator Camera…")
                    .tint(.white)
                    .foregroundStyle(.white)
            }
            #else
            CameraPreviewView(previewLayer: cameraManager.previewLayer)
                .ignoresSafeArea()
            #endif

            VStack {
                HStack {
                    Button {
                        onCancel()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundStyle(.white)
                            .padding()
                    }
                    Spacer()
                }

                Spacer()

                if cameraManager.permissionDenied {
                    Text("Camera access denied. Enable it in Settings.")
                        .foregroundStyle(.white)
                        .padding()
                        .background(.black.opacity(0.6))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .padding(.bottom, 40)
                } else {
                    Button {
                        if let image = cameraManager.capturePhoto() {
                            onCapture(image)
                        }
                    } label: {
                        Circle()
                            .fill(.white)
                            .frame(width: 72, height: 72)
                            .overlay(
                                Circle()
                                    .stroke(.white.opacity(0.6), lineWidth: 4)
                                    .frame(width: 86, height: 86)
                            )
                    }
                    .padding(.bottom, 40)
                }
            }
        }
        .onAppear { cameraManager.startSession() }
        .onDisappear { cameraManager.stopSession() }
    }
}

private struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> PreviewContainerView {
        let view = PreviewContainerView()
        view.previewLayer = previewLayer
        return view
    }

    func updateUIView(_ uiView: PreviewContainerView, context: Context) {}

    final class PreviewContainerView: UIView {
        var previewLayer: AVCaptureVideoPreviewLayer? {
            didSet { if let previewLayer { layer.addSublayer(previewLayer) } }
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer?.frame = bounds
        }
    }
}
