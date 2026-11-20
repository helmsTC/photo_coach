plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "2.0.20" apply false
    id("org.jetbrains.kotlin.jvm") version "2.0.20" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.20" apply false
    id("org.jetbrains.kotlin.plugin.serialization") version "2.0.20" apply false
}

// Optional: build outside the source tree (e.g. when the checkout sits on a slow or shared drive).
//   APERTUREIQ_BUILD=~/.cache/apertureiq ./gradlew :app:assembleDebug
System.getenv("APERTUREIQ_BUILD")?.let { root ->
    allprojects { layout.buildDirectory.set(file("$root/${project.name}")) }
}
