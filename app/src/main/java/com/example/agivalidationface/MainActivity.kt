package com.example.agivalidationface

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.agivalidationface.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.pow
import kotlin.math.sqrt
class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var faceDetector: FaceDetector
    private lateinit var tflite: Interpreter
    private var imageCapture: ImageCapture? = null

    // Armazenamento simples para o embedding do rosto cadastrado
    private var registeredFaceEmbedding: FloatArray? = null

    // Limiar de similaridade para considerar os rostos como da mesma pessoa
    private val SIMILARITY_THRESHOLD = 0.8f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 1. Verificar permissões da câmera
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // 2. Inicializar o ML Kit Face Detector
        val highAccuracyOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()
        faceDetector = FaceDetection.getClient(highAccuracyOpts)

        // 3. Carregar o modelo do TensorFlow Lite
        try {
            tflite = Interpreter(loadModelFile())
        } catch (e: Exception) {
            Log.e("MainActivity", "Erro ao carregar modelo TFLite", e)
            Toast.makeText(this, "Não foi possível carregar o modelo de reconhecimento.", Toast.LENGTH_LONG).show()
        }

        // 4. Configurar listeners dos botões
        binding.btnRegister.setOnClickListener { takePhoto(isRegistering = true) }
        binding.btnValidate.setOnClickListener { takePhoto(isRegistering = false) }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder().build()
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (exc: Exception) {
                Log.e("MainActivity", "Falha ao iniciar a câmera", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto(isRegistering: Boolean) {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback(), ImageCapture.OnImageSavedCallback {
                override fun onCaptureSuccess(image: androidx.camera.core.ImageProxy) {
                    val bitmap = image.toBitmap()
                    detectFace(bitmap, isRegistering)
                    image.close() // Importante fechar o proxy
                }

                override fun onError(exc: ImageCaptureException) {
                    Log.e("MainActivity", "Erro ao capturar foto: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {} // Não usado aqui
            }
        )
    }

    private fun detectFace(bitmap: Bitmap, isRegistering: Boolean) {
        val image = InputImage.fromBitmap(bitmap, 0)
        faceDetector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    binding.tvStatus.text = "Nenhum rosto detectado!"
                    return@addOnSuccessListener
                }

                // Pega o primeiro rosto detectado
                val face = faces.first()
                val faceBitmap = cropFace(bitmap, face.boundingBox)

                if (faceBitmap == null) {
                    binding.tvStatus.text = "Erro ao processar rosto."
                    return@addOnSuccessListener
                }

                val embedding = getFaceEmbedding(faceBitmap)

                if (isRegistering) {
                    registeredFaceEmbedding = embedding
                    binding.ivProfile.setImageBitmap(faceBitmap)
                    binding.tvStatus.text = "Rosto cadastrado com sucesso!"
                } else {
                    validateFace(embedding)
                }
            }
            .addOnFailureListener { e ->
                Log.e("MainActivity", "Falha na detecção de rosto", e)
                binding.tvStatus.text = "Erro ao detectar rosto."
            }
    }

    private fun validateFace(newEmbedding: FloatArray) {
        val storedEmbedding = registeredFaceEmbedding
        if (storedEmbedding == null) {
            binding.tvStatus.text = "Nenhum rosto cadastrado. Cadastre primeiro."
            return
        }

        val similarity = cosineSimilarity(storedEmbedding, newEmbedding)
        Log.d("MainActivity", "Similaridade: $similarity")

        if (similarity > SIMILARITY_THRESHOLD) {
            binding.tvStatus.text = "Validação bem-sucedida! (Similaridade: %.2f)".format(similarity)
        } else {
            binding.tvStatus.text = "Falha na validação. Rosto diferente. (Similaridade: %.2f)".format(similarity)
        }
    }

    private fun cropFace(bitmap: Bitmap, boundingBox: Rect): Bitmap? {
        // Garante que o corte não saia dos limites do bitmap
        val left = maxOf(0, boundingBox.left)
        val top = maxOf(0, boundingBox.top)
        val width = if (left + boundingBox.width() > bitmap.width) bitmap.width - left else boundingBox.width()
        val height = if (top + boundingBox.height() > bitmap.height) bitmap.height - top else boundingBox.height()

        if (width <= 0 || height <= 0) return null

        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }

    private fun getFaceEmbedding(bitmap: Bitmap): FloatArray {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true)
        val byteBuffer = convertBitmapToByteBuffer(resizedBitmap)

        val output = Array(1) { FloatArray(512) } // Ajuste o tamanho (192 ou 512) conforme o seu modelo!

        tflite.run(byteBuffer, output)
        return output[0]
    }

    // Funções auxiliares (Helpers)

    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd("mobilefacenet.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * 160 * 160 * 3) // MUDOU
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(160 * 160) // MUDOU
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0
        for (i in 0 until 160) { // MUDOU
            for (j in 0 until 160) { // MUDOU
                val value = intValues[pixel++]
                // Normaliza os pixels para o intervalo [-1, 1] que o modelo espera
                byteBuffer.putFloat(((value shr 16 and 0xFF) - 127.5f) / 128.0f)
                byteBuffer.putFloat(((value shr 8 and 0xFF) - 127.5f) / 128.0f)
                byteBuffer.putFloat(((value and 0xFF) - 127.5f) / 128.0f)
            }
        }
        return byteBuffer
    }

    private fun cosineSimilarity(x: FloatArray, y: FloatArray): Float {
        var dotProduct = 0.0f
        var normX = 0.0f
        var normY = 0.0f
        for (i in x.indices) {
            dotProduct += x[i] * y[i]
            normX += x[i].pow(2)
            normY += y[i].pow(2)
        }
        return dotProduct / (sqrt(normX) * sqrt(normY))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissões não concedidas.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}