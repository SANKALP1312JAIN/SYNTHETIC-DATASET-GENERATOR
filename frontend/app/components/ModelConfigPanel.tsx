"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { useAppContext } from "../context/AppContext"
import { Switch } from "@/components/ui/switch"
import { Loader2, Play, Download, Info, Heart, DollarSign, ShoppingCart, Settings, Factory, Stethoscope, BarChart2 } from "lucide-react"

export default function ModelConfigPanel() {
  const { modelConfig, setModelConfig, noCodeMode, setNoCodeMode, uploadedFile } = useAppContext()
  const [isTraining, setIsTraining] = useState(false)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [ganType, setGanType] = useState<string | null>(null)
  const [dataType, setDataType] = useState<string | null>(null)
  const router = useRouter()

  const updateConfig = (key: keyof typeof modelConfig, value: any) => {
    setModelConfig({ ...modelConfig, [key]: value })
  }

  const handleTrainModel = async () => {
    if (!uploadedFile) {
      alert("Please upload a dataset first")
      return
    }

    setIsTraining(true)
    setDownloadUrl(null)

    try {
      // Convert the uploaded data back to CSV format
      const csvContent = [
        uploadedFile.columns.join(','),
        ...uploadedFile.data.map(row => 
          uploadedFile.columns.map(col => row[col]).join(',')
        )
      ].join('\n')

      // Create a file blob
      const fileBlob = new Blob([csvContent], { type: 'text/csv' })
      const file = new File([fileBlob], uploadedFile.name, { type: 'text/csv' })

      // Create form data
      const formData = new FormData()
      formData.append('file', file)
      formData.append('domain', modelConfig.domain || 'custom')
      formData.append('privacy_level', modelConfig.privacyLevel || 'medium')
      formData.append('num_samples', modelConfig.numSamples?.toString() || '1000')
      formData.append('output_format', 'csv')

      // Send training request
      const response = await fetch('http://localhost:8000/generate-synthetic-data', {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const result = await response.json()
        if (result.download_url) {
          // Always use backend port for download
          setDownloadUrl("http://localhost:8000" + result.download_url)
          if (result.gan_type) {
            setGanType(result.gan_type)
          } else if (result.details && result.details.requirements && result.details.requirements.gan_type) {
            setGanType(result.details.requirements.gan_type)
          } else {
            setGanType(null)
          }
          if (result.data_type) {
            setDataType(result.data_type)
          } else {
            setDataType(null)
          }
        } else {
          alert('Training succeeded but no download link was provided.')
        }
      } else {
        const error = await response.json()
        alert(`Generation failed: ${error.error}`)
      }
    } catch (error) {
      console.error('Training error:', error)
      alert('Error during training. Please try again.')
    } finally {
      setIsTraining(false)
    }
  }

  const prebuiltRecipes = [
    "Quick Synthetic Data",
    "High Quality Generation",
    "Privacy Preserving",
    "Balanced Dataset",
    "Anomaly Detection Ready",
  ]

  // Domain templates with icons and colors
  const domainTemplates = [
    {
      id: "healthcare",
      name: "Healthcare",
      description: "Patient records, medical data, clinical trials",
      icon: Stethoscope,
      color: "bg-red-500",
    },
    {
      id: "finance",
      name: "Finance",
      description: "Transaction data, customer profiles, risk metrics",
      icon: DollarSign,
      color: "bg-green-500",
    },
    {
      id: "retail",
      name: "Retail",
      description: "Customer behavior, sales data, inventory",
      icon: ShoppingCart,
      color: "bg-blue-500",
    },
    {
      id: "manufacturing",
      name: "Manufacturing",
      description: "Process, defect, and sensor data",
      icon: Factory,
      color: "bg-yellow-600",
    },
    {
      id: "research",
      name: "Research",
      description: "Academic, scientific, or experimental data",
      icon: BarChart2,
      color: "bg-purple-500",
    },
    {
      id: "custom",
      name: "Custom",
      description: "Configure your own domain-specific template",
      icon: Settings,
      color: "bg-gray-500",
    },
  ]

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-medium text-gray-900">Model Configuration</h3>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">No-code Mode</span>
          <Switch checked={noCodeMode} onCheckedChange={setNoCodeMode} />
        </div>
      </div>

      {noCodeMode ? (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Choose Recipe</label>
            <select className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500">
              {prebuiltRecipes.map((recipe, index) => (
                <option key={index} value={recipe}>
                  {recipe}
                </option>
              ))}
            </select>
          </div>
          <button 
            onClick={handleTrainModel}
            disabled={isTraining || !uploadedFile}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isTraining ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Generate Synthetic Data
              </>
            )}
          </button>
        </div>
      ) : (
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Model Type</label>
            <div className="flex space-x-4">
              {["GAN", "VAE"].map((model) => (
                <label key={model} className="flex items-center">
                  <input
                    type="radio"
                    name="model"
                    value={model}
                    checked={modelConfig.model === model}
                    onChange={(e) => updateConfig("model", e.target.value)}
                    className="mr-2"
                  />
                  {model}
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Batch Size: {modelConfig.batchSize}</label>
            <input
              type="range"
              min="8"
              max="128"
              step="8"
              value={modelConfig.batchSize}
              onChange={(e) => updateConfig("batchSize", Number.parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Latent Noise: {modelConfig.latentNoise}
            </label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={modelConfig.latentNoise}
              onChange={(e) => updateConfig("latentNoise", Number.parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Epochs: {modelConfig.epochs}</label>
            <input
              type="range"
              min="50"
              max="500"
              step="10"
              value={modelConfig.epochs}
              onChange={(e) => updateConfig("epochs", Number.parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Privacy Level</label>
            <select
              value={modelConfig.privacyLevel || 'medium'}
              onChange={e => updateConfig("privacyLevel", e.target.value)}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Number of Samples: {modelConfig.numSamples}</label>
            <input
              type="number"
              min="1"
              value={modelConfig.numSamples || 1000}
              onChange={e => updateConfig("numSamples", Number.parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <button 
            onClick={handleTrainModel}
            disabled={isTraining || !uploadedFile}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isTraining ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Training Model...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Train Model
              </>
            )}
          </button>
          {downloadUrl && (
            <a
              href={downloadUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="w-full mt-2 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors flex items-center justify-center text-center"
              style={{ textDecoration: 'none' }}
            >
              <Download className="h-4 w-4 mr-2" />
              Download Synthetic Data
            </a>
          )}
          {(ganType || dataType) && (
            <div className="w-full mt-2 bg-blue-600 text-white py-2 px-4 rounded-md flex items-center justify-center text-center">
              <Info className="h-4 w-4 mr-2" />
              <span>
                {ganType && <>GAN used: <b>{ganType}</b></>}
                {ganType && dataType && <span className="mx-2">|</span>}
                {dataType && <>Data type: <b>{dataType}</b></>}
              </span>
            </div>
          )}
          
          {!uploadedFile && (
            <p className="text-sm text-gray-500 text-center">
              Please upload a dataset first to train the model
            </p>
          )}
        </div>
      )}
    </div>
  )
}
