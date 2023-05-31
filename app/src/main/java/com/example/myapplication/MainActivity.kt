package com.example.myapplication

import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val button : Button = findViewById(R.id.button)
        val textView : TextView = findViewById(R.id.textView)

        val assetManager: AssetManager = assets
        val interpreter: Interpreter = loadModel(assetManager)
        // Dummy input, not normalized yet. Click button to see the prediction
        val features = featureNormalization(83460F, 22380F, 92F, 23132.5F, 168.9F) // Quality = 77
        val result = targetNormalizationInverse(performInference(interpreter, features))
        button.setOnClickListener{
            textView.text = result.toString()
        }
    }
    // Load the TensorFlow Lite model
    private fun loadModel(assetManager: AssetManager): Interpreter {
        val modelFilename = "model.tflite"
        val modelFileDescriptor: AssetFileDescriptor = assetManager.openFd(modelFilename)
        val inputStream = FileInputStream(modelFileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset: Long = modelFileDescriptor.startOffset
        val declaredLength: Long = modelFileDescriptor.declaredLength
        val modelByteBuffer: MappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        return Interpreter(modelByteBuffer)
    }

    // Perform inference using the TensorFlow Lite model
    private fun performInference(interpreter: Interpreter, features: FloatArray): Float {
        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()

        val inputBuffer = ByteBuffer.allocateDirect(inputShape[0] * inputShape[1] * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        for (feature in features) {
            inputBuffer.putFloat(feature)
        }
        val outputBuffer = ByteBuffer.allocateDirect(outputShape[0] * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)
        return outputBuffer.getFloat(0)
    }
    // normalization func
    private fun featureNormalization(start:Float,
                                     end:Float,
                                     regularity:Float,
                                     timeAsleep:Float,
                                     timeBeforeSleep:Float) : FloatArray {
        val startNorm = (start-21780)/(94380-21780)
        val endNorm = (end-3900)/(84540-3900)
        val regularityNorm = (regularity+1)/(100+1)
        val timeAsleepNorm = (timeAsleep/45769.4)
        val timeBeforeSleepNorm = (timeBeforeSleep-560)/(5677.7-560)

        return floatArrayOf(startNorm, endNorm, regularityNorm, timeAsleepNorm.toFloat(), timeBeforeSleepNorm.toFloat())
    }
    private fun targetNormalizationInverse(quality:Float):Float{
        return quality * (100-7) + 7
    }
}