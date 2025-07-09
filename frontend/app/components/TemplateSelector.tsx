"use client"

import { useAppContext } from "../context/AppContext"
import { Heart, DollarSign, ShoppingCart, Settings, Factory, Stethoscope, BarChart2 } from "lucide-react"
import { useState } from "react"

const templates = [
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

export default function TemplateSelector() {
  const { selectedTemplate, setSelectedTemplate, setModelConfig } = useAppContext()
  const [apiMessage, setApiMessage] = useState<string>("")

  const handleTemplateSelect = async (templateId: string) => {
    setSelectedTemplate(templateId)

    // Update model config based on template
    const templateConfigs = {
      healthcare: { batchSize: 16, latentNoise: 0.05, epochs: 150, model: "VAE" as const },
      finance: { batchSize: 32, latentNoise: 0.1, epochs: 200, model: "GAN" as const },
      retail: { batchSize: 64, latentNoise: 0.15, epochs: 100, model: "GAN" as const },
      manufacturing: { batchSize: 48, latentNoise: 0.12, epochs: 120, model: "GAN" as const },
      research: { batchSize: 24, latentNoise: 0.08, epochs: 180, model: "VAE" as const },
      custom: { batchSize: 32, latentNoise: 0.1, epochs: 100, model: "GAN" as const },
    }

    setModelConfig(templateConfigs[templateId as keyof typeof templateConfigs])

    // Fetch message from FastAPI backend
    try {
      const res = await fetch(`http://localhost:8001/api/message/${templateId}`)
      if (res.ok) {
        const data = await res.json()
        setApiMessage(data.message)
      } else {
        setApiMessage("Failed to fetch message from backend.")
      }
    } catch (err) {
      // Do not show any error message if backend is unreachable
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Domain Templates</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {templates.map((template) => (
          <div
            key={template.id}
            onClick={() => handleTemplateSelect(template.id)}
            className={`
              border-2 rounded-lg p-4 cursor-pointer transition-all
              ${
                selectedTemplate === template.id
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-200 hover:border-gray-300"
              }
            `}
          >
            <div className="flex items-start space-x-3">
              <div className={`p-2 rounded-lg ${template.color}`}>
                <template.icon className={`h-6 w-6 ${template.id === 'manufacturing' ? 'text-yellow-400' : 'text-white'}`} />
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-gray-900">{template.name}</h4>
                <p className="text-sm text-gray-500 mt-1">{template.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
      {apiMessage && (
        <div className="mt-4 p-3 bg-blue-100 text-blue-800 rounded">
          {apiMessage}
        </div>
      )}
    </div>
  )
}
