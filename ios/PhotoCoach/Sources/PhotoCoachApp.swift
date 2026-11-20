import SwiftUI

@main
struct PhotoCoachApp: App {
    init() {
        if Diagnostics.requested {
            Task.detached { await Diagnostics.run() }
        }
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .preferredColorScheme(.dark)
        }
    }
}
