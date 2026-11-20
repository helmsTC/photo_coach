# ONNX Runtime looks its Java classes up from native code
-keep class ai.onnxruntime.** { *; }
# kotlinx.serialization: keep generated serializers
-keepattributes *Annotation*, InnerClasses
-keepclassmembers class **$$serializer { *; }
-keepclasseswithmembers class com.mosaiclens.apertureiq.** { kotlinx.serialization.KSerializer serializer(...); }
-keep,includedescriptorclasses class com.mosaiclens.apertureiq.**$$serializer { *; }
