# Media service

This directory will contain the single Go2 WebRTC owner for camera frames,
microphone input, and bark audio. It is deliberately documentation-only in the
initial scaffold so the new repository cannot connect to Woof accidentally.

The service contract must expose monotonically increasing frame IDs, source
timestamps, connection generations, and explicit freshness health.
