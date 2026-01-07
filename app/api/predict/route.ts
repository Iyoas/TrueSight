export const runtime = "nodejs";

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.TRUESIGHT_BACKEND_URL || "http://localhost:8000";

/**
 * Proxy endpoint that forwards the uploaded image to the Python backend.
 *
 * Frontend usage:
 *
 * const formData = new FormData();
 * formData.append("file", file);
 * await fetch("/api/predict", {
 *   method: "POST",
 *   body: formData,
 * });
 */
export async function POST(req: NextRequest) {
  try {
    const incomingFormData = await req.formData();
    const file = incomingFormData.get("file");

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json(
        { error: "No image file provided under form field 'file'." },
        { status: 400 }
      );
    }

    // Forward the file to the Python backend
    const formData = new FormData();
    // @ts-ignore - Next.js Blob may not have name typed, but it's present at runtime
    const fileName = (file as any).name || "upload.png";
    formData.append("file", file, fileName);

    const backendResponse = await fetch(
      `${BACKEND_URL}/predict_with_explainability`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!backendResponse.ok) {
      const text = await backendResponse.text();
      return NextResponse.json(
        {
          error: "Backend returned an error.",
          status: backendResponse.status,
          details: text,
        },
        { status: 500 }
      );
    }

    const data = await backendResponse.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error: any) {
    console.error("[/api/predict] Error:", error);
    return NextResponse.json(
      {
        error: "Failed to process prediction request.",
        details: error?.message ?? String(error),
      },
      { status: 500 }
    );
  }
}
