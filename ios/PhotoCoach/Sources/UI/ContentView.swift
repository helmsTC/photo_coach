import PhotosUI
import SwiftUI

enum Palette {
    static let a = Color(red: 0x39 / 255, green: 0x87 / 255, blue: 0xE5 / 255)      // photo A: blue
    static let b = Color(red: 0xD9 / 255, green: 0x59 / 255, blue: 0x26 / 255)      // photo B: orange
    static let card = Color(white: 0.11)
    static let line = Color(white: 0.22)
}

@MainActor
final class CompareModel: ObservableObject {
    @Published var dataA: Data?
    @Published var dataB: Data?
    @Published var thumbA: UIImage?
    @Published var thumbB: UIImage?
    @Published var busy = false
    @Published var error: String?
    @Published var result: Comparison?

    func set(_ data: Data?, isA: Bool) {
        let img = data.flatMap { UIImage(data: $0) }
        if isA { dataA = data; thumbA = img } else { dataB = data; thumbB = img }
        result = nil
        error = nil
    }

    func run() {
        guard let dataA, let dataB, !busy else { return }
        busy = true
        error = nil
        Task {
            do {
                result = try await PhotoJudge.shared.compare(dataA: dataA, dataB: dataB, nameA: "My photo", nameB: "Reference")
            } catch {
                self.error = error.localizedDescription
            }
            busy = false
        }
    }
}

struct ContentView: View {
    @StateObject private var model = CompareModel()
    @State private var showSelfTest = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Pick two photos. A twin network judges which is stronger, shows why, and tells you what to change to close the gap.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 12) {
                        PhotoSlot(title: "My photo", tint: Palette.a, image: model.thumbA) { model.set($0, isA: true) }
                        PhotoSlot(title: "Reference", tint: Palette.b, image: model.thumbB) { model.set($0, isA: false) }
                    }
                    Button(action: model.run) {
                        HStack {
                            if model.busy { ProgressView().tint(.black) }
                            Text(model.busy ? "Analysing…" : "Compare").fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.white)
                    .foregroundStyle(.black)
                    .disabled(model.dataA == nil || model.dataB == nil || model.busy)

                    if let e = model.error {
                        Text(e).font(.footnote).foregroundStyle(.red)
                    }
                    if let r = model.result {
                        ResultView(r: r)
                    }
                }
                .padding(16)
            }
            .background(Color(white: 0.07).ignoresSafeArea())
            .navigationTitle("Photo Coach")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Self-test") { showSelfTest = true }.font(.footnote)
                }
            }
            .sheet(isPresented: $showSelfTest) { SelfTestView() }
        }
    }
}

struct PhotoSlot: View {
    let title: String
    let tint: Color
    let image: UIImage?
    let onPick: (Data?) -> Void
    @State private var item: PhotosPickerItem?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Circle().fill(tint).frame(width: 9, height: 9)
                Text(title).font(.subheadline.weight(.semibold))
            }
            PhotosPicker(selection: $item, matching: .images) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10).fill(Palette.card)
                    if let image {
                        Image(uiImage: image).resizable().scaledToFill()
                    } else {
                        VStack(spacing: 6) {
                            Image(systemName: "photo.badge.plus").font(.title2)
                            Text("Choose").font(.footnote)
                        }
                        .foregroundStyle(.secondary)
                    }
                }
                .frame(height: 150)
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Palette.line))
            }
            .onChange(of: item) { _, new in
                guard let new else { return }
                Task {
                    let data = try? await new.loadTransferable(type: Data.self)
                    await MainActor.run { onPick(data) }
                }
            }
        }
    }
}
