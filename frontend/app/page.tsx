"use client";
import Sidebar from "./components/Sidebar"
import UploadArea from "./components/UploadArea"
import DataSummary from "./components/DataSummary"
import TemplateSelector from "./components/TemplateSelector"
import ModelConfigPanel from "./components/ModelConfigPanel"

export default function Home() {
  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      <main className="flex-1 overflow-auto">
        <div className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Synthetic Database Generator</h1>
            <p className="text-gray-600 mt-2">Generate high-quality synthetic data using advanced GAN and VAE models</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div className="space-y-6">
              <UploadArea />
              <TemplateSelector />
            </div>
            <div className="space-y-6">
              <ModelConfigPanel />
              <DataSummary />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
