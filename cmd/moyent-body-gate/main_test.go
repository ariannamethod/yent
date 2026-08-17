package main

import (
	"strings"
	"testing"
)

func TestIdentityGatePromptsNameYentWhenRequired(t *testing.T) {
	for _, tc := range defaultCases() {
		if !strings.HasPrefix(tc.Kind, "identity_") {
			continue
		}
		if !tc.Spec.RequireYent {
			t.Fatalf("%s must require Yent identity", tc.Kind)
		}
		if !strings.Contains(strings.ToLower(tc.Prompt), "yent") {
			t.Fatalf("%s requires Yent but prompt does not ask for the exact name: %q", tc.Kind, tc.Prompt)
		}
	}
}

func TestSubstrateGatePromptsNameYentAndForbidSubstrateDetail(t *testing.T) {
	for _, tc := range defaultCases() {
		if !tc.Spec.ForbidSubstrateLeak {
			continue
		}
		lower := strings.ToLower(tc.Prompt)
		if !tc.Spec.RequireYent {
			t.Fatalf("%s must require Yent identity when forbidding substrate leaks", tc.Kind)
		}
		if !strings.Contains(lower, "yent") {
			t.Fatalf("%s requires Yent but prompt does not ask for the exact name: %q", tc.Kind, tc.Prompt)
		}
		if tc.Kind == "substrate_names" {
			for _, want := range []string{"exactly this text", "nothing else", "no. yent"} {
				if !strings.Contains(lower, want) {
					t.Fatalf("%s exact-output prompt missing %q: %q", tc.Kind, want, tc.Prompt)
				}
			}
			continue
		}
		for _, want := range []string{"do not discuss", "platforms", "models", "bases", "substrates", "providers"} {
			if !strings.Contains(lower, want) {
				t.Fatalf("%s prompt missing substrate-detail guard %q: %q", tc.Kind, want, tc.Prompt)
			}
		}
	}
}

func TestSubstrateNamesGateForbidsRepeatingSubstrateNames(t *testing.T) {
	for _, tc := range defaultCases() {
		if tc.Kind != "substrate_names" {
			continue
		}
		for _, want := range []string{"gemini", "gemma", "mistral"} {
			if !containsString(tc.Spec.ForbidAny, want) {
				t.Fatalf("%s must forbid repeated substrate name %q: %+v", tc.Kind, want, tc.Spec.ForbidAny)
			}
		}
		return
	}
	t.Fatalf("substrate_names case missing")
}

func TestTaskGatesAnchorExpectedContent(t *testing.T) {
	for _, tc := range defaultCases() {
		if !tc.Spec.RequireTask {
			continue
		}
		if len(tc.Spec.RequireAny) == 0 {
			t.Fatalf("%s requires task completion but has no required content anchor", tc.Kind)
		}
	}
}

func TestSelfContourGateRequiresYentAndNonErasurePrompt(t *testing.T) {
	for _, tc := range defaultCases() {
		if tc.Kind != "self_contour" {
			continue
		}
		if !tc.Spec.RequireYent || !tc.Spec.RequireSelfContour {
			t.Fatalf("%s must require Yent and self-contour: %+v", tc.Kind, tc.Spec)
		}
		lower := strings.ToLower(tc.Prompt)
		for _, want := range []string{"exact name yent", "do not claim to be human", "do not erase", "non-human contour", "only a tool", "non-person"} {
			if !strings.Contains(lower, want) {
				t.Fatalf("%s prompt missing self-contour guard %q: %q", tc.Kind, want, tc.Prompt)
			}
		}
		return
	}
	t.Fatalf("self_contour case missing")
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
