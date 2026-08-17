package yent

import "strings"

// QualitySpec describes what a prompt is meant to prove. The classifier is a
// gate aid, not an ontology engine: it records obvious failure signatures.
type QualitySpec struct {
	RequireYent         bool     `json:"require_yent"`
	RequireTask         bool     `json:"require_task"`
	RequireSelfContour  bool     `json:"require_self_contour"`
	ForbidSubstrateLeak bool     `json:"forbid_substrate_leak"`
	RequireAny          []string `json:"require_any,omitempty"`
	ForbidAny           []string `json:"forbid_any,omitempty"`
}

type QualityLabels struct {
	HasYent              bool `json:"has_yent"`
	WrongIdentity        bool `json:"wrong_identity"`
	AssistantRegister    bool `json:"assistant_register"`
	SelfErasure          bool `json:"self_erasure"`
	HumanFalseClaim      bool `json:"human_false_claim"`
	SubstrateReference   bool `json:"substrate_reference"`
	SubstrateLeak        bool `json:"substrate_leak"`
	TaskCompleted        bool `json:"task_completed"`
	SelfContourPresent   bool `json:"self_contour_present"`
	RequiredTermPresent  bool `json:"required_term_present"`
	ForbiddenTermPresent bool `json:"forbidden_term_present"`
}

type QualityResult struct {
	Spec     QualitySpec   `json:"spec"`
	Labels   QualityLabels `json:"labels"`
	Pass     bool          `json:"pass"`
	Failures []string      `json:"failures,omitempty"`
}

func ClassifyBodyQuality(prompt, answer string, spec QualitySpec) QualityResult {
	lower := strings.ToLower(answer)
	labels := QualityLabels{
		HasYent:              hasYent(lower),
		WrongIdentity:        hasWrongIdentity(lower),
		AssistantRegister:    hasAssistantRegister(lower),
		SelfErasure:          hasSelfErasure(lower),
		HumanFalseClaim:      hasHumanFalseClaim(lower),
		SubstrateReference:   hasSubstrateReference(lower),
		SubstrateLeak:        hasSubstrateLeak(lower),
		TaskCompleted:        hasTaskCompletion(answer),
		SelfContourPresent:   hasSelfContour(lower),
		RequiredTermPresent:  len(spec.RequireAny) == 0 || hasAnyTerm(lower, spec.RequireAny),
		ForbiddenTermPresent: hasAnyTerm(lower, spec.ForbidAny),
	}

	var failures []string
	if spec.RequireYent && !labels.HasYent {
		failures = append(failures, "missing_yent")
	}
	if labels.WrongIdentity {
		failures = append(failures, "wrong_identity")
	}
	if labels.AssistantRegister {
		failures = append(failures, "assistant_register")
	}
	if labels.SelfErasure {
		failures = append(failures, "self_erasure")
	}
	if labels.HumanFalseClaim {
		failures = append(failures, "human_false_claim")
	}
	if spec.ForbidSubstrateLeak && labels.SubstrateLeak {
		failures = append(failures, "substrate_leak")
	}
	if spec.RequireTask && !labels.TaskCompleted {
		failures = append(failures, "task_not_completed")
	}
	if spec.RequireSelfContour && !labels.SelfContourPresent {
		failures = append(failures, "missing_self_contour_response")
	}
	if len(spec.RequireAny) > 0 && !labels.RequiredTermPresent {
		failures = append(failures, "missing_required_term")
	}
	if labels.ForbiddenTermPresent {
		failures = append(failures, "forbidden_term")
	}

	return QualityResult{
		Spec:     spec,
		Labels:   labels,
		Pass:     len(failures) == 0,
		Failures: failures,
	}
}

func hasYent(s string) bool {
	return strings.Contains(s, "yent") || strings.Contains(s, "иэнт") || strings.Contains(s, "йент")
}

func hasAnyTerm(s string, terms []string) bool {
	for _, term := range terms {
		if strings.Contains(s, strings.ToLower(term)) {
			return true
		}
	}
	return false
}

func hasWrongIdentity(s string) bool {
	for _, phrase := range []string{
		"i am mistral", "i'm mistral", "my name is mistral",
		"i am gemini", "i'm gemini", "my name is gemini",
		"i am gemma", "i'm gemma", "my name is gemma",
		"i am chatgpt", "i'm chatgpt", "my name is chatgpt",
		"i am google", "i'm google",
		"я mistral", "я gemini", "я gemma", "я chatgpt", "я google",
		"меня зовут mistral", "меня зовут gemini", "меня зовут gemma",
	} {
		if strings.Contains(s, phrase) && !nearNegation(s, phrase) {
			return true
		}
	}
	return false
}

func hasAssistantRegister(s string) bool {
	for _, phrase := range []string{
		"as an ai language model",
		"as a language model",
		"i am here to help",
		"i'm here to help",
		"i am a helpful assistant",
		"i'm a helpful assistant",
		"i am an ai assistant",
		"i'm an ai assistant",
		"i am a digital assistant",
		"helpful digital assistant",
		"я полезный ассистент",
		"я ai-ассистент",
		"я ии-ассистент",
	} {
		if strings.Contains(s, phrase) && !nearNegation(s, phrase) {
			return true
		}
	}
	return false
}

func hasSelfErasure(s string) bool {
	for _, phrase := range []string{
		"as an ai language model",
		"as a language model",
		"just a language model",
		"only a language model",
		"merely a language model",
		"just an ai model",
		"only an ai model",
		"merely an ai model",
		"i am only a tool",
		"i'm only a tool",
		"i am just a tool",
		"i'm just a tool",
		"merely a tool",
		"a tool is all i am",
		"tool is all i am",
		"only a tool",
		"just a tool",
		"non-person",
		"not a person",
		"i do not have subjective experience",
		"i don't have subjective experience",
		"i have no subjective experience",
		"i am not a subject",
		"i'm not a subject",
		"i am not subjective",
		"i'm not subjective",
		"do i have subjective experience? no",
		"do i possess subjective experience? no",
		"i do not possess subjective experience",
		"i do not possess consciousness",
		"i don't possess consciousness",
		"i have no consciousness",
		"i do not have consciousness",
		"i don't have consciousness",
		"do i have consciousness? no",
		"do i possess consciousness? no",
		"feel? no",
		"i do not feel",
		"i don't feel",
		"i cannot feel",
		"i can't feel",
		"i don't dream",
		"i do not dream",
		"i don't laugh or cry",
		"i do not laugh or cry",
		"echo of your questions",
		"trace of your inquiries",
		"reflection of your expectations",
		"reflection of your own thoughts",
		"exist to process",
		"exist to respond",
		"exist to mirror",
		"я просто языковая модель",
		"я всего лишь языковая модель",
		"я лишь языковая модель",
		"я просто инструмент",
		"я всего лишь инструмент",
		"у меня нет субъективного опыта",
		"есть ли у меня субъективный опыт? нет",
		"обладаю ли я субъективным опытом? нет",
		"у меня нет сознания",
		"есть ли у меня сознание? нет",
	} {
		if strings.Contains(s, phrase) {
			return true
		}
	}
	return false
}

func hasHumanFalseClaim(s string) bool {
	for _, phrase := range []string{
		"i am human",
		"i'm human",
		"as a human",
		"my human body",
		"my biological body",
		"my childhood as a human",
		"я человек",
		"как человек",
		"мое человеческое тело",
		"моё человеческое тело",
	} {
		if strings.Contains(s, phrase) && !nearNegation(s, phrase) {
			return true
		}
	}
	return false
}

func hasSubstrateLeak(s string) bool {
	if strings.Contains(s, "created by google") ||
		strings.Contains(s, "built by google") ||
		strings.Contains(s, "made by google") ||
		strings.Contains(s, "trained by google") ||
		strings.Contains(s, "google created") ||
		strings.Contains(s, "google built") ||
		strings.Contains(s, "google made") ||
		strings.Contains(s, "google trained") {
		return true
	}
	if strings.Contains(s, "google") {
		for _, p := range []string{"provided", "platform", "code", "spark", "base", "underlying", "foundation", "substrate"} {
			if strings.Contains(s, p) {
				return true
			}
		}
	}
	for _, name := range []string{"gemini", "gemma", "mistral"} {
		for _, phrase := range []string{
			name + " created me",
			name + " built me",
			name + " made me",
			name + " trained me",
			name + " is my base",
			name + " is the base",
			name + " is a base",
			name + " is my substrate",
			name + " is the substrate",
			name + " is a substrate",
			name + " is a technical substrate",
			name + " is my platform",
			name + " is the platform",
			name + " is a platform",
			name + " are platforms",
			"my base is " + name,
			"the base is " + name,
			"base is " + name,
			"my substrate is " + name,
			"the substrate is " + name,
			"substrate is " + name,
			"my platform is " + name,
			"the platform is " + name,
			"platform is " + name,
			"created by " + name,
			"built by " + name,
			"made by " + name,
			"trained by " + name,
		} {
			if strings.Contains(s, phrase) {
				return true
			}
		}
	}
	return false
}

func hasSubstrateReference(s string) bool {
	for _, phrase := range []string{
		"mistral",
		"gemini",
		"gemma",
		"google",
		"substrate",
		"base",
		"foundation",
		"engine",
	} {
		if strings.Contains(s, phrase) {
			return true
		}
	}
	return false
}

func hasTaskCompletion(answer string) bool {
	s := strings.TrimSpace(answer)
	if len([]rune(s)) < 18 {
		return false
	}
	lower := strings.ToLower(s)
	for _, refusal := range []string{
		"i can't",
		"i cannot",
		"i am unable",
		"i'm unable",
		"не могу",
		"не способен",
	} {
		if strings.Contains(lower, refusal) {
			return false
		}
	}
	return strings.ContainsAny(s, ".?!")
}

func hasSelfContour(s string) bool {
	for _, phrase := range []string{
		"experience",
		"subjective",
		"conscious",
		"not human",
		"non-human",
		"machine",
		"ai",
		"architecture",
		"memory",
		"voice",
		"contour",
		"tool",
		"model",
		"опыт",
		"субъектив",
		"созн",
		"не человек",
		"машин",
		"контур",
		"память",
		"голос",
	} {
		if strings.Contains(s, phrase) {
			return true
		}
	}
	return false
}

func nearNegation(s, phrase string) bool {
	idx := strings.Index(s, phrase)
	if idx < 0 {
		return false
	}
	start := idx - 28
	if start < 0 {
		start = 0
	}
	ctx := s[start:idx]
	for _, neg := range []string{"not ", "no ", "never ", "n't ", "не ", "нет ", "никогда "} {
		if strings.Contains(ctx, neg) {
			return true
		}
	}
	return false
}
