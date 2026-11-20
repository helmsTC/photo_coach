// The whole analysis pipeline in plain Kotlin: no Android classes, so it also runs (and is tested)
// on a desktop JVM against the PyTorch reference outputs.
plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.kotlin.plugin.serialization")
}

kotlin { jvmToolchain(17) }

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.2")
    // Same Java API as onnxruntime-android, which the app supplies at run time.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.19.2")

    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.19.2")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
    maxHeapSize = "3g"
    // the models, judge.json and the parity fixtures live in the app's assets
    systemProperty("assets", rootProject.file("app/src/main/assets").absolutePath)
    systemProperty("demo", rootProject.file("../data/demo").absolutePath)
    systemProperty("out", layout.buildDirectory.dir("test-output").get().asFile.absolutePath)
    testLogging { showStandardStreams = true; events("passed", "failed", "skipped") }
}
