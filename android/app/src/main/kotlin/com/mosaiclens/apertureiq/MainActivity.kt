package com.mosaiclens.apertureiq

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import com.mosaiclens.apertureiq.ui.AppModel
import com.mosaiclens.apertureiq.ui.DebugOptions
import com.mosaiclens.apertureiq.ui.RootView
import com.mosaiclens.apertureiq.ui.Theme
import com.mosaiclens.apertureiq.ui.Tier

class MainActivity : ComponentActivity() {
    private val model: AppModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge(SystemBarStyle.dark(android.graphics.Color.TRANSPARENT), SystemBarStyle.dark(android.graphics.Color.TRANSPARENT))
        super.onCreate(savedInstanceState)
        if (BuildConfig.DEBUG) {
            // Launch options for screenshots and testing, e.g.
            //   adb shell am start -n com.mosaiclens.apertureiq/.MainActivity --ei autosample 0 --es tab coach
            val x = intent?.extras
            if (x != null) {
                model.applyDebug(
                    DebugOptions(
                        noSplash = x.getBoolean("nosplash", false),
                        tier = if (x.getBoolean("free", false)) Tier.FREE else if (x.getBoolean("pro", false)) Tier.PRO else null,
                        autoSample = if (x.containsKey("autosample")) x.getInt("autosample") else null,
                        tab = x.getString("tab"),
                        lens = x.getString("lens"),
                        photo = if (x.containsKey("photo")) x.getInt("photo") else null,
                        selfTest = x.getBoolean("selftest", false),
                        expandAll = x.getBoolean("expandall", false),
                    ),
                )
            }
        }
        setContent {
            MaterialTheme(colorScheme = darkColorScheme(primary = Theme.a, surface = Theme.bg, background = Theme.bg)) {
                RootView(model)
            }
        }
    }
}
