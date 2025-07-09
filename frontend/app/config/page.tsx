import Sidebar from "../components/Sidebar"
import ModelConfigPanel from "../components/ModelConfigPanel"

export default function ConfigPage() {
  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      <main className="flex-1 overflow-auto">
        <div className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Model Configuration</h1>
            <p className="text-gray-600 mt-2">
              Fine-tune your GAN/VAE model parameters for optimal synthetic data generation
            </p>
          </div>

          <div className="max-w-2xl">
            <ModelConfigPanel />
          </div>
        </div>
      </main>
    </div>
  )
}
