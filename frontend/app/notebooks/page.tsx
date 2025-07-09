"use client"

import { useState } from "react"
import Sidebar from "../components/Sidebar"
import { BookOpen, Download, Play, Code } from "lucide-react"
import { useAppContext } from "../context/AppContext"

export default function NotebooksPage() {
  const { uploadedFile, selectedTemplate, modelConfig } = useAppContext()
  const [isGenerating, setIsGenerating] = useState(false)

  const generateNotebook = async () => {
    setIsGenerating(true)

    // Mock API call to generate notebook
    try {
      await new Promise((resolve) => setTimeout(resolve, 2000)) // Simulate processing

      // Create mock notebook content
      const notebookContent = {
        cells: [
          {
            cell_type: "markdown",
            source: [`# Synthetic Data Generation Notebook\n\nGenerated for ${selectedTemplate} domain`],
          },
          {
            cell_type: "code",
            source: [
              "import pandas as pd\n",
              "import numpy as np\n",
              "from sklearn.preprocessing import StandardScaler\n",
              `# Model configuration: ${JSON.stringify(modelConfig, null, 2)}\n`,
            ],
          },
        ],
      }

      const blob = new Blob([JSON.stringify(notebookContent, null, 2)], {
        type: "application/json",
      })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "synthetic_data_generation.ipynb"
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error("Notebook generation failed:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  const notebookTemplates = [
    {
      name: "Basic Synthetic Data Generation",
      description: "Simple GAN/VAE implementation for synthetic data",
      complexity: "Beginner",
    },
    {
      name: "Advanced Privacy-Preserving Generation",
      description: "Differential privacy with synthetic data generation",
      complexity: "Advanced",
    },
    {
      name: "Domain-Specific Data Synthesis",
      description: "Customized generation for your specific domain",
      complexity: "Intermediate",
    },
    {
      name: "Quality Assessment & Validation",
      description: "Comprehensive evaluation of synthetic data quality",
      complexity: "Intermediate",
    },
  ]

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      <main className="flex-1 overflow-auto">
        <div className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Jupyter Notebooks</h1>
            <p className="text-gray-600 mt-2">
              Generate ready-to-use Python notebooks for your synthetic data projects
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Generate Custom Notebook</h3>

              <div className="space-y-4 mb-6">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-medium text-gray-900 mb-2">Current Configuration</h4>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p>
                      <strong>Dataset:</strong> {uploadedFile?.name || "No file uploaded"}
                    </p>
                    <p>
                      <strong>Template:</strong> {selectedTemplate}
                    </p>
                    <p>
                      <strong>Model:</strong> {modelConfig.model}
                    </p>
                    <p>
                      <strong>Batch Size:</strong> {modelConfig.batchSize}
                    </p>
                    <p>
                      <strong>Epochs:</strong> {modelConfig.epochs}
                    </p>
                  </div>
                </div>
              </div>

              <button
                onClick={generateNotebook}
                disabled={isGenerating || !uploadedFile}
                className={`
                  w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-md
                  ${
                    isGenerating || !uploadedFile ? "bg-gray-300 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
                  } text-white transition-colors
                `}
              >
                {isGenerating ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4" />
                    <span>Generate & Download Notebook</span>
                  </>
                )}
              </button>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Notebook Templates</h3>

              <div className="space-y-3">
                {notebookTemplates.map((template, index) => (
                  <div key={index} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900">{template.name}</h4>
                        <p className="text-sm text-gray-600 mt-1">{template.description}</p>
                      </div>
                      <span
                        className={`
                        px-2 py-1 text-xs rounded-full
                        ${
                          template.complexity === "Beginner"
                            ? "bg-green-100 text-green-800"
                            : template.complexity === "Intermediate"
                              ? "bg-yellow-100 text-yellow-800"
                              : "bg-red-100 text-red-800"
                        }
                      `}
                      >
                        {template.complexity}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 mt-3">
                      <button className="flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800">
                        <Play className="h-3 w-3" />
                        <span>Preview</span>
                      </button>
                      <button className="flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800">
                        <Code className="h-3 w-3" />
                        <span>View Code</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-6 bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Notebook Features</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-start space-x-3">
                <BookOpen className="h-6 w-6 text-blue-500 mt-1" />
                <div>
                  <h4 className="font-medium text-gray-900">Complete Documentation</h4>
                  <p className="text-sm text-gray-600">Detailed explanations and comments</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <Code className="h-6 w-6 text-green-500 mt-1" />
                <div>
                  <h4 className="font-medium text-gray-900">Production Ready</h4>
                  <p className="text-sm text-gray-600">Optimized and tested code</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <Play className="h-6 w-6 text-purple-500 mt-1" />
                <div>
                  <h4 className="font-medium text-gray-900">Interactive Examples</h4>
                  <p className="text-sm text-gray-600">Runnable code with sample data</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
