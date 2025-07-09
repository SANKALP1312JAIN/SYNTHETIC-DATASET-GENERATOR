"use client"

import { createContext, useContext, useState, type ReactNode } from "react"

interface UploadedFile {
  name: string
  size: number
  data: any[]
  columns: string[]
}

interface ModelConfig {
  batchSize: number
  latentNoise: number
  epochs: number
  model: "GAN" | "VAE"
  domain?: string
  privacyLevel?: string
  numSamples?: number
}

interface AppContextType {
  uploadedFile: UploadedFile | null
  setUploadedFile: (file: UploadedFile | null) => void
  selectedTemplate: string
  setSelectedTemplate: (template: string) => void
  modelConfig: ModelConfig
  setModelConfig: (config: ModelConfig) => void
  noCodeMode: boolean
  setNoCodeMode: (mode: boolean) => void
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export function AppProvider({ children }: { children: ReactNode }) {
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null)
  const [selectedTemplate, setSelectedTemplate] = useState("custom")
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    batchSize: 32,
    latentNoise: 0.1,
    epochs: 100,
    model: "GAN",
    domain: "custom",
    privacyLevel: "medium",
    numSamples: 1000,
  })
  const [noCodeMode, setNoCodeMode] = useState(false)

  return (
    <AppContext.Provider
      value={{
        uploadedFile,
        setUploadedFile,
        selectedTemplate,
        setSelectedTemplate,
        modelConfig,
        setModelConfig,
        noCodeMode,
        setNoCodeMode,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useAppContext() {
  const context = useContext(AppContext)
  if (context === undefined) {
    throw new Error("useAppContext must be used within an AppProvider")
  }
  return context
}
