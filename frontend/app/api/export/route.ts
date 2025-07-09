import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { format, data } = await request.json()

    let content: string
    let contentType: string

    switch (format) {
      case "csv":
        if (data.length > 0) {
          const headers = Object.keys(data[0]).join(",")
          const rows = data.map((row: any) => Object.values(row).join(","))
          content = [headers, ...rows].join("\n")
        } else {
          content = "No data available"
        }
        contentType = "text/csv"
        break

      case "json":
        content = JSON.stringify(data, null, 2)
        contentType = "application/json"
        break

      case "parquet":
        // Mock parquet content (in real implementation, use a parquet library)
        content = JSON.stringify(data, null, 2)
        contentType = "application/octet-stream"
        break

      default:
        return NextResponse.json({ error: "Unsupported format" }, { status: 400 })
    }

    return new NextResponse(content, {
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": `attachment; filename="synthetic_data.${format}"`,
      },
    })
  } catch (error) {
    console.error("Export error:", error)
    return NextResponse.json({ error: "Export failed" }, { status: 500 })
  }
}
