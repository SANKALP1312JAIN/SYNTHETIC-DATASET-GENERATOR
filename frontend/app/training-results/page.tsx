"use client"

import { useEffect, useState } from "react"
import { useSearchParams } from "next/navigation"
import Sidebar from "../components/Sidebar"
import { Download, CheckCircle, AlertCircle, Loader2, Info, Warning, XCircle } from "lucide-react"

interface TrainingResult {
  job_id: string
  status: string
  message: string
  download_url: string
  data_info: {
    original_shape: [number, number]
    synthetic_shape: [number, number]
    num_samples: number
    preprocessing_info?: {
      original_shape: [number, number]
      missing_values: Record<string, number>
      categorical_columns: string[]
      numerical_columns: string[]
      preprocessing_steps: string[]
      warnings: string[]
      final_shape: [number, number]
    }
  }
  progress?: string
  errors?: string[]
  warnings?: string[]
}

export default function TrainingResultsPage() {
  const searchParams = useSearchParams()
  const jobId = searchParams.get("job_id")
  const [result, setResult] = useState<TrainingResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (jobId) {
      fetchTrainingResult(jobId)
    }
  }, [jobId])

  const fetchTrainingResult = async (id: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/job-status/${id}`)
      if (response.ok) {
        const data = await response.json()
        setResult({
          job_id: id,
          status: data.status,
          message: data.status === "completed" ? "Model training completed successfully" : data.progress || "Processing",
          download_url: `http://localhost:8001/api/download/${id}`,
          data_info: {
            original_shape: data.data_shape,
            synthetic_shape: data.synthetic_shape,
            num_samples: data.num_samples,
            preprocessing_info: data.preprocessing_info
          },
          progress: data.progress,
          errors: data.errors || [],
          warnings: data.warnings || []
        })
      } else {
        setError("Failed to fetch training result")
      }
    } catch (err) {
      setError("Error connecting to backend")
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async () => {
    if (!result) return
    
    try {
      const response = await fetch(result.download_url)
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `synthetic_data_${result.job_id}.csv`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (err) {
      console.error("Download failed:", err)
    }
  }

  if (loading) {
    return (
      <div className="flex h-screen bg-gray-100">
        <Sidebar />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-500" />
            <p className="text-gray-600">Loading training results...</p>
          </div>
        </main>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-screen bg-gray-100">
        <Sidebar />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <AlertCircle className="h-8 w-8 mx-auto mb-4 text-red-500" />
            <p className="text-red-600">{error}</p>
          </div>
        </main>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="flex h-screen bg-gray-100">
        <Sidebar />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <p className="text-gray-600">No training result found</p>
          </div>
        </main>
      </div>
    )
  }

  const preprocessingInfo = result.data_info.preprocessing_info

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <div className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Training Results</h1>
            <p className="text-gray-600 mt-2">
              {result.status === "completed" 
                ? "Your synthetic data has been generated successfully" 
                : result.progress || "Processing training results"}
            </p>
          </div>

          <div className="max-w-6xl space-y-6">
            {/* Status and Progress */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center mb-4">
                {result.status === "completed" ? (
                  <CheckCircle className="h-6 w-6 text-green-500 mr-2" />
                ) : result.status === "failed" ? (
                  <XCircle className="h-6 w-6 text-red-500 mr-2" />
                ) : (
                  <Loader2 className="h-6 w-6 animate-spin text-blue-500 mr-2" />
                )}
                <h2 className="text-xl font-semibold text-gray-900">
                  {result.status === "completed" ? "Training Completed" : 
                   result.status === "failed" ? "Training Failed" : "Training in Progress"}
                </h2>
              </div>
              
              {result.progress && (
                <p className="text-gray-600 mb-4">{result.progress}</p>
              )}

              {/* Errors */}
              {result.errors && result.errors.length > 0 && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                  <div className="flex items-center mb-2">
                    <XCircle className="h-5 w-5 text-red-500 mr-2" />
                    <h3 className="font-medium text-red-900">Errors</h3>
                  </div>
                  <ul className="text-sm text-red-800 space-y-1">
                    {result.errors.map((error, index) => (
                      <li key={index}>• {error}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Warnings */}
              {result.warnings && result.warnings.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                  <div className="flex items-center mb-2">
                    <Warning className="h-5 w-5 text-yellow-500 mr-2" />
                    <h3 className="font-medium text-yellow-900">Warnings</h3>
                  </div>
                  <ul className="text-sm text-yellow-800 space-y-1">
                    {result.warnings.map((warning, index) => (
                      <li key={index}>• {warning}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Data Information */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Data Information</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Original Data</h4>
                  <p className="text-sm text-gray-600">
                    Shape: {result.data_info.original_shape[0]} rows × {result.data_info.original_shape[1]} columns
                  </p>
                </div>
                
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Synthetic Data</h4>
                  <p className="text-sm text-gray-600">
                    Shape: {result.data_info.synthetic_shape[0]} rows × {result.data_info.synthetic_shape[1]} columns
                  </p>
                  <p className="text-sm text-gray-600">
                    Samples generated: {result.data_info.num_samples}
                  </p>
                </div>
              </div>

              {/* Download Button */}
              {result.status === "completed" && (
                <div className="border-t pt-4">
                  <button
                    onClick={handleDownload}
                    className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download Synthetic Data
                  </button>
                </div>
              )}
            </div>

            {/* Preprocessing Information */}
            {preprocessingInfo && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Data Preprocessing Details</h3>
                
                {/* Column Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div className="bg-blue-50 rounded-lg p-4">
                    <h4 className="font-medium text-blue-900 mb-2">Numerical Columns</h4>
                    <p className="text-sm text-blue-800">
                      {preprocessingInfo.numerical_columns.length} columns
                    </p>
                    {preprocessingInfo.numerical_columns.length > 0 && (
                      <p className="text-xs text-blue-700 mt-1">
                        {preprocessingInfo.numerical_columns.join(', ')}
                      </p>
                    )}
                  </div>
                  
                  <div className="bg-green-50 rounded-lg p-4">
                    <h4 className="font-medium text-green-900 mb-2">Categorical Columns</h4>
                    <p className="text-sm text-green-800">
                      {preprocessingInfo.categorical_columns.length} columns
                    </p>
                    {preprocessingInfo.categorical_columns.length > 0 && (
                      <p className="text-xs text-green-700 mt-1">
                        {preprocessingInfo.categorical_columns.join(', ')}
                      </p>
                    )}
                  </div>
                </div>

                {/* Missing Values */}
                {Object.keys(preprocessingInfo.missing_values).length > 0 && (
                  <div className="bg-orange-50 rounded-lg p-4 mb-6">
                    <h4 className="font-medium text-orange-900 mb-2">Missing Values Handled</h4>
                    <div className="text-sm text-orange-800">
                      {Object.entries(preprocessingInfo.missing_values).map(([col, count]) => (
                        <p key={col}>• {col}: {count} missing values</p>
                      ))}
                    </div>
                  </div>
                )}

                {/* Preprocessing Steps */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Preprocessing Steps</h4>
                  <ol className="text-sm text-gray-700 space-y-1">
                    {preprocessingInfo.preprocessing_steps.map((step, index) => (
                      <li key={index} className="flex items-start">
                        <span className="text-gray-500 mr-2">{index + 1}.</span>
                        <span>{step}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              </div>
            )}

            {/* Next Steps */}
            {result.status === "completed" && (
              <div className="bg-blue-50 rounded-lg p-4">
                <h3 className="font-medium text-blue-900 mb-2">What's Next?</h3>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• Download your synthetic data using the button above</li>
                  <li>• Use the synthetic data for testing, development, or analysis</li>
                  <li>• The data maintains the statistical properties of your original dataset</li>
                  <li>• Return to the main page to generate more synthetic data</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
} 