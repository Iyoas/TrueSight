export const runtime = "nodejs";

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.TRUESIGHT_BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const payload = await req.json();

    const backendResponse = await fetch(
      `${BACKEND_URL}/explain_cues_batch`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
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
    console.error("[/api/explain-cues-batch] Error:", error);
    return NextResponse.json(
      {
        error: "Failed to process batch cue explanation request.",
        details: error?.message ?? String(error),
      },
      { status: 500 }
    );
  }
}
