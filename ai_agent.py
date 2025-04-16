import json
import uuid
import re
import os
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from crewai import Agent, Task, Crew
from pydantic import ValidationError
from typing import Dict, List, Tuple, Optional
from profile_schema import UserProfile, ContactMethod, GeneralInfo, Transportation, Retail, Healthcare, Legal, Hospitality, Education, Entertainment, RealEstate, Fitness, TravelPlanning, Interaction
from typing import get_origin
import google.generativeai as genai

# Set up Gemini API
# Note: In a real implementation, use environment variables for API keys
API_KEY = "AIzaSyBIaK3s4ody4j9nlOuJkuwNtwWkVrdtppE"  # Replace with actual API key or use environment variables
genai.configure(api_key=API_KEY)

# Mapping of domain names to their respective model classes
DOMAIN_CLASSES = {
    "transportation": Transportation,
    "retail": Retail,
    "healthcare": Healthcare,
    "legal": Legal,
    "hospitality": Hospitality,
    "education": Education,
    "entertainment": Entertainment,
    "real_estate": RealEstate,
    "fitness": Fitness,
    "travel_planning": TravelPlanning,
}

# Define a mapping of common sub-domains for better understanding in the LLM prompt
SUB_DOMAINS = {
    "transportation": ["flights", "trains", "buses", "taxis", "car_rentals", "cruises", "public_transit"],
    "retail": ["clothing", "electronics", "groceries", "furniture", "online_shopping", "luxury_goods"],
    "healthcare": ["general_checkup", "specialist_visit", "emergency_care", "telehealth", "mental_health", "dental_care"],
    "legal": ["family_law", "criminal_defense", "estate_planning", "business_law", "immigration", "intellectual_property"],
    "hospitality": ["restaurants", "cafes", "bars", "hotels", "resorts", "catering", "food_delivery"],
    "education": ["k12", "university", "online_courses", "tutoring", "professional_development", "language_learning"],
    "entertainment": ["movies", "concerts", "theater", "sports_events", "streaming_services", "gaming", "theme_parks"],
    "real_estate": ["buying", "selling", "renting", "investment_property", "commercial_real_estate", "property_management"],
    "fitness": ["gym_membership", "personal_training", "group_classes", "home_workout", "sports_leagues", "outdoor_activities"],
    "travel_planning": ["vacation_packages", "business_travel", "adventure_travel", "family_trips", "solo_travel", "international_travel"],
}


class LLMDomainDetector:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def detect_domain_and_subdomain(self, prompt: str) -> Tuple[Optional[str], Optional[str], float]:
        try:
            # Create a structured prompt for the LLM
            llm_prompt = f"""
            Analyze the following user request and identify the most relevant domain and sub-domain.
            
            User request: "{prompt}"
            
            Available domains and their sub-domains:
            {json.dumps(SUB_DOMAINS, indent=2)}
            
            Return ONLY a JSON object with this format:
            {{
                "domain": "[domain name]",
                "sub_domain": "[sub_domain name]",
                "confidence": [0-1 value representing confidence]
            }}
            
            If you're unsure, set confidence below 0.1. Use only domains and sub-domains from the provided lists.
            """
            
            response = self.model.generate_content(llm_prompt)
            response_text = response.text
            
            # Extract JSON from response
            json_str = response_text.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()
                
            result = json.loads(json_str)
            
            domain = result.get("domain")
            sub_domain = result.get("sub_domain")
            confidence = result.get("confidence", 0.0)
            
            # Validate domain and sub_domain
            if domain in SUB_DOMAINS and sub_domain in SUB_DOMAINS.get(domain, []):
                return domain, sub_domain, confidence
            else:
                # Fallback if response doesn't match our structure
                return domain if domain in SUB_DOMAINS else None, None, 0.0
                
        except Exception as e:
            print(f"Error in LLM domain detection: {e}")
            return None, None, 0.0


class QuestionGenerator:
    def __init__(self):
        self.choice_fields = {
            "trip_type": ["one-way", "round-trip"],
            "seat_preference": ["window", "aisle", "no preference"],
            "delivery_preference": ["delivery", "pickup"],
            "appointment_type": ["check-up", "consultation", "procedure"],
            "consultation_mode": ["in-person", "virtual"],
            "dining_style": ["casual", "fine dining", "takeout"],
            "learning_mode": ["online", "in-person"],
            "event_type": ["one-time", "subscription"],
            "purchase_type": ["buy", "rent", "sell"],
            "trainer_preference": ["trainer", "group"],
            "preferred_contact_method": [e.value for e in ContactMethod]
        }
        self.phrase_variations = {
            "preferred": ["What are your preferred", "Which do you prefer as your", "What's your preferred"],
            "categories": ["Which categories do you enjoy", "What types of categories do you like"],
            "brands": ["Which brands do you prefer", "What are your favorite brands for"],
            "budget": ["What's your budget range for", "What's your budget for"],
            "frequency": ["How often do you", "How regularly do you engage in"],
            "type": ["What kind of", "Which style of"],
            "loyalty": ["Which loyalty programs are you part of", "Do you have any favorite memberships for"],
            "quality": ["How important is quality to you for", "What's your take on the quality of"],
            "preference": ["What's your preference for", "How do you like your"],
        }
        self.llm = genai.GenerativeModel('gemini-2.0-flash')

    def get_creative_phrase(self, field_name: str, field_type: str) -> str:
        import random
        
        # Format the field name for display by replacing underscores with spaces
        display_name = field_name.replace('_', ' ')
        
        # Check for keywords in field_name to pick a relevant phrase
        for key in self.phrase_variations:
            if key in field_name:
                phrase = random.choice(self.phrase_variations[key])
                # For list fields, make it clear we want multiple items
                if field_type == "list":
                    return f"{phrase} {display_name}, separated by commas"
                return f"{phrase} {display_name}"
                
        # Default phrases for different field types
        if field_type == "list":
            return f"What are your preferred {display_name}, separated by commas"
        
        # Generic fallback that still includes the field name
        return f"What's your {display_name}"

    def generate_questions(self, domain: str, fields: Dict, prefix: str = "", sub_domain: str = None, user_prompt: str = None) -> List[Dict]:
        # First, generate standard questions from the schema
        standard_questions = []
        for field_name, field_info in fields.items():
            question_id = f"{prefix}{field_name}" if prefix else field_name
            field_type = self.infer_field_type(field_info)

            if field_type == "nested":
                nested_model = field_info.annotation
                if hasattr(nested_model, "model_fields"):
                    nested_fields = nested_model.model_fields
                    standard_questions.extend(self.generate_questions(domain, nested_fields, f"{question_id}."))
            elif field_type == "list":
                phrase = self.get_creative_phrase(field_name, "list")
                question_text = f"{phrase}?"
                standard_questions.append({"id": question_id, "text": question_text, "type": "list"})
            else:
                phrase = self.get_creative_phrase(field_name, "text")
                question_text = f"{phrase}?"
                if field_name in self.choice_fields:
                    question_text += f" (Options: {', '.join(self.choice_fields[field_name])})"
                standard_questions.append({"id": question_id, "text": question_text, "type": field_type})
        
        # If we have sub_domain and user_prompt, enhance the questions with LLM
        if sub_domain and user_prompt:
            return self.enhance_questions_with_llm(standard_questions, domain, sub_domain, user_prompt)
        
        return standard_questions

    def enhance_questions_with_llm(self, standard_questions: List[Dict], domain: str, sub_domain: str, user_prompt: str) -> List[Dict]:
        try:
            # Prepare the prompt for the LLM
            questions_str = "\n".join([f"- {q['id']}: {q['text']}" for q in standard_questions])
            
            llm_prompt = f"""
            I need to generate targeted questions for a user based on their specific request.
            
            User's request: "{user_prompt}"
            Domain: {domain}
            Sub-domain: {sub_domain}
            
            Here are the standard questions from our schema:
            {questions_str}
            
            Please improve these questions to make them more specific to the user's request and the sub-domain.
            Prioritize the most relevant questions for their specific need, and phrase them in a conversational, context-aware way.
            
            For example, if they're booking a movie ticket, focus on movie preferences, time, location, etc.
            If they're looking for a doctor's appointment, focus on symptoms, specialty needs, urgency, etc.
            
            Return ONLY a JSON array with this format:
            [
              {{"id": "field_name", "text": "Improved question text?", "type": "text|list", "priority": 1}},
              ...
            ]
            
            Include a 'priority' field (1 is highest) to indicate question importance for this specific request.
            Keep the same 'id' values as the original questions.
            Limit to 10 most relevant questions.
            """
            
            response = self.llm.generate_content(llm_prompt)
            response_text = response.text
            
            # Extract JSON from response
            json_str = response_text.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()
                
            enhanced_questions = json.loads(json_str)
            
            # Sort by priority
            enhanced_questions.sort(key=lambda x: x.get('priority', 999))
            
            # Remove priority field before returning
            for q in enhanced_questions:
                if 'priority' in q:
                    del q['priority']
            
            # If we got empty or invalid results, fallback to standard questions
            if not enhanced_questions:
                return standard_questions
                
            return enhanced_questions
                
        except Exception as e:
            print(f"Error enhancing questions with LLM: {e}")
            return standard_questions

    # Fix the infer_field_type method to handle Pydantic v2
    def infer_field_type(self, field_info):
        from typing import get_origin
        annotation = field_info.annotation
        
        # Check if it's a list type
        if get_origin(annotation) is list:
            return "list"
        # Check if it's a Pydantic model (nested)
        elif hasattr(annotation, "model_fields"):
            return "nested"
        
        return "text"


class ProfileManager:
    def __init__(self, profile_file="profile.json"):
        self.profile_file = profile_file
        self.profile = self.load_profile()

    def load_profile(self):
        try:
            with open(self.profile_file, 'r') as f:
                data = json.load(f)
                # Extract domains data and remove it from the main data
                domains_data = data.pop('domains', {})
                # Create UserProfile instance without domains
                profile = UserProfile(**data)
                # Manually instantiate specific domain models from raw data
                for domain, cls in DOMAIN_CLASSES.items():
                    if domain in domains_data:
                        profile.domains[domain] = cls(**domains_data[domain])
                return profile
        except (FileNotFoundError, ValidationError, ValueError):
            # Return a default profile if loading fails
            return UserProfile(
                user_id=str(uuid.uuid4()),
                general_info=GeneralInfo(),
                domains={domain: cls() for domain, cls in DOMAIN_CLASSES.items()},
                interaction_history=[]
            )

    def save_profile(self):
        # Convert the profile to a dictionary
        profile_dict = self.profile.model_dump()
        
        # Convert domain objects to dictionaries
        domains_dict = {}
        for domain_name, domain_obj in self.profile.domains.items():
            domains_dict[domain_name] = domain_obj.model_dump()
        
        # Replace the domains in the profile dictionary
        profile_dict['domains'] = domains_dict
        
        # Save to file
        with open(self.profile_file, 'w') as f:
            json.dump(profile_dict, f, indent=2)

    def update_general_info(self, key: str, value):
        if "." in key:
            parent, child = key.split(".")
            parent_obj = getattr(self.profile.general_info, parent)
            setattr(parent_obj, child, value)
        else:
            setattr(self.profile.general_info, key, value)
        self.save_profile()

    def update_domain_info(self, domain: str, updates: dict):
        # Get the domain object
        domain_obj = self.profile.domains[domain]
        
        # Update each field in the domain object
        for key, value in updates.items():
            # Handle follow-up responses
            if key.startswith("follow_up_"):
                # Ensure follow_up_data exists
                if not hasattr(domain_obj, 'follow_up_data'):
                    domain_obj.follow_up_data = {}
                domain_obj.follow_up_data[key] = value
            # Handle nested fields (with dots in the key)
            elif "." in key:
                parts = key.split(".")
                current = domain_obj
                # Navigate to the nested object
                for part in parts[:-1]:
                    current = getattr(current, part)
                # Set the value on the nested object
                setattr(current, parts[-1], value)
            else:
                # Set the value directly on the domain object if the field exists
                if key in domain_obj.__class__.model_fields:
                    setattr(domain_obj, key, value)
                else:
                    print(f"Warning: Field '{key}' not found in {domain} schema")
        
        # Save the updated profile
        self.save_profile()
        
    def log_interaction(self, domain: str, query: str, response: str, sub_domain: str = None):
        interaction = Interaction(
            domain=domain,
            timestamp=datetime.now().isoformat(),
            query=query,
            response={"value": response}
        )
        
        # If there's a sub_domain, add it to the interaction data
        if sub_domain:
            interaction.response["sub_domain"] = sub_domain
            
        self.profile.interaction_history.append(interaction)
        self.save_profile()


class AIAgent:
    def __init__(self):
        self.profile_manager = ProfileManager()
        self.question_generator = QuestionGenerator()
        self.domain_detector = LLMDomainDetector()
        self.session = PromptSession(style=Style.from_dict({
            'prompt': 'fg:ansiblue bold',
            '': 'fg:ansigreen'
        }))
        self.setup_crew()

    def setup_crew(self):
        schema_parser = Agent(
            role='Schema Parser',
            goal='Analyze the Pydantic schema to extract fields and types for question generation',
            backstory='Expert in data modeling and schema analysis, skilled at interpreting complex structures.',
            verbose=True
        )
        question_crafter = Agent(
            role='Question Crafter',
            goal='Generate creative, context-aware questions based on schema fields',
            backstory='Creative linguist with a knack for crafting engaging, domain-specific questions.',
            verbose=True
        )

        self.parse_task = Task(
            description='Parse the Pydantic schema for a given domain and return field metadata.',
            agent=schema_parser,
            expected_output='Dictionary of field names, types, and metadata'
        )
        self.craft_task = Task(
            description='Generate at least 15 creative questions for the domain based on field metadata.',
            agent=question_crafter,
            expected_output='List of question dictionaries with id, text, and type'
        )

        self.crew = Crew(
            agents=[schema_parser, question_crafter],
            tasks=[self.parse_task, self.craft_task],
            verbose=True
        )

    def start(self):
        print("Welcome! I'm your AI assistant. Let's build your profile.")
        self.collect_general_info()
        self.start_conversation()

    def collect_general_info(self):
        questions = [
            {"id": "name", "text": "What's your full name?"},
            {"id": "email", "text": "What's your email address?"},
            {"id": "phone", "text": "What's your phone number?"},
            {"id": "location.city", "text": "Which city do you live in?"},
            {"id": "location.country", "text": "Which country do you live in?"},
            {"id": "preferred_contact_method", "text": f"How do you prefer to be contacted? (Options: {', '.join([e.value for e in ContactMethod])})"}
        ]
        for q in questions:
            response = self.session.prompt(q["text"] + " ")
            self.profile_manager.update_general_info(q["id"], response)

    def start_conversation(self):
        print("\nHow can I help you today? Type 'exit' to quit.")
        while True:
            # Get user prompt
            user_prompt = self.session.prompt("You: ")
            
            # Check if user wants to exit
            if user_prompt.lower() == 'exit':
                print("Thank you for using the AI agent. Goodbye!")
                break
                
            # Detect domain and sub-domain from user prompt using LLM
            domain, sub_domain, confidence = self.domain_detector.detect_domain_and_subdomain(user_prompt)
            
            if domain and confidence >= 0.5:
                domain_display = domain.replace('_', ' ').title()
                sub_domain_display = sub_domain.replace('_', ' ').title() if sub_domain else "General"
                
                print(f"\nI see you're interested in our {domain_display} services - specifically {sub_domain_display}!")
                self.handle_domain_questions(domain, user_prompt, sub_domain)
            elif domain:
                print(f"\nI think you might be interested in our {domain.replace('_', ' ').title()} services, but I'm not entirely sure.")
                confirmation = self.session.prompt(f"Is {domain.replace('_', ' ').title()} what you're looking for? (yes/no): ")
                
                if confirmation.lower() in ['yes', 'y']:
                    self.handle_domain_questions(domain, user_prompt)
                else:
                    self.show_domain_selection_menu()
            else:
                print("I'm not sure what service you're looking for. Could you be more specific?")
                self.show_domain_selection_menu()

    def show_domain_selection_menu(self):
        print("Here are the available services:")
        domains = list(self.profile_manager.profile.domains.keys())
        for i, domain in enumerate(domains, 1):
            print(f"{i}. {domain.title().replace('_', ' ')}")
        
        # Allow manual selection as fallback
        choice = self.session.prompt("Please select a service (1-{}) or press Enter to try again: ".format(len(domains)))
        if choice:
            try:
                choice = int(choice)
                if 1 <= choice <= len(domains):
                    self.handle_domain_questions(domains[choice - 1], "")
                else:
                    print("Invalid choice. Let's try again.")
            except ValueError:
                print("Let's try again with more specific information.")

    def handle_domain_questions(self, domain: str, original_prompt: str, sub_domain: str = None):
        domain_model = self.profile_manager.profile.domains[domain]
        
        # Generate questions based on domain, sub_domain, and user prompt
        questions = self.question_generator.generate_questions(
            domain, 
            domain_model.__class__.model_fields,
            sub_domain=sub_domain,
            user_prompt=original_prompt
        )
        
        responses = {}
        
        domain_display = domain.replace('_', ' ').title()
        sub_domain_display = sub_domain.replace('_', ' ').title() if sub_domain else ""
        
        print(f"\nI'll help you with your {domain_display}{' - '+sub_domain_display if sub_domain else ''} request. Let me gather some information:")
        
        # Log the original interaction
        self.profile_manager.log_interaction(domain, original_prompt, "Processing request", sub_domain)
        
        # Collect user answers for the most relevant questions (prioritized by LLM)
        for q in questions[:15]:  # Limit to top questions to avoid overwhelming
            answer = self.session.prompt(q["text"] + " ")
            field_id = q["id"]
            if answer.strip():  # Only store non-empty answers
                responses[field_id] = answer.strip()
        
        # Process responses for list fields
        field_info = domain_model.__class__.model_fields
        processed_responses = {}
        
        for key, value in responses.items():
            if "." in key:
                # Handle nested fields
                parent, child = key.split(".", 1)
                if parent in field_info:
                    processed_responses[key] = value
            elif key in field_info:
                field = field_info[key]
                # Handle list fields
                if get_origin(field.annotation) is list and value:
                    processed_responses[key] = [item.strip() for item in value.split(",") if item.strip()]
                else:
                    processed_responses[key] = value
        
        # Update the domain in the profile manager
        self.profile_manager.update_domain_info(domain, processed_responses)
        
        # Generate follow-up questions based on the collected information
        follow_up_questions = self.generate_follow_up_questions(domain, processed_responses, sub_domain, original_prompt)
        
        print("\nThank you for providing that information!")
        print("Let me ask you some follow-up questions to better understand your needs:")
        
        # Ask the follow-up questions interactively
        follow_up_responses = {}
        for i, question in enumerate(follow_up_questions, 1):
            answer = self.session.prompt(f"{i}. {question} ")
            if answer.strip().lower() != 'skip':
                follow_up_responses[f"follow_up_{i}"] = answer.strip()
        
        # Update profile with follow-up responses
        if follow_up_responses:
            self.profile_manager.update_domain_info(domain, follow_up_responses)
        
        # Generate final response
        final_response = self.generate_final_response(domain, {**processed_responses, **follow_up_responses}, sub_domain)
        print("\n" + final_response)
        
        # Log the final interaction
        self.profile_manager.log_interaction(domain, "Conversation complete", final_response, sub_domain)

    def generate_follow_up_questions(self, domain: str, responses: dict, sub_domain: str = None, original_prompt: str = None) -> List[str]:
        """Generate personalized follow-up questions based on collected information"""
        try:
            responses_str = json.dumps(responses, indent=2)
            
            llm_prompt = f"""
            Based on the following context, generate 15 highly personalized follow-up questions that:
            - Build upon the information already collected
            - Are focused on the user's specific needs
            - Help gather more detailed or insightful information
            - Are relevant to {domain} {f"and {sub_domain}" if sub_domain else ""}
            
            Original Request: "{original_prompt}"
            Collected Information:
            {responses_str}
            
            Return ONLY a numbered list of questions, one per line.
            """
            
            llm = genai.GenerativeModel('gemini-2.0-flash')
            response = llm.generate_content(llm_prompt)
            questions = response.text.strip().split('\n')
            
            # Clean up the questions (remove numbers if present)
            cleaned_questions = []
            for q in questions:
                # Remove leading numbers and dots
                cleaned = re.sub(r'^\d+\.\s*', '', q).strip()
                if cleaned:
                    cleaned_questions.append(cleaned)
            
            return cleaned_questions[:15]  # Return max 15 questions
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return []

    def generate_final_response(self, domain: str, all_responses: dict, sub_domain: str = None) -> str:
        """Generate a concluding response based on all collected information"""
        # You can implement this similarly to your current generate_domain_response
        # but with a more concluding tone
        return f"Thank you for all the information! I now have a good understanding of your {domain} {f'and {sub_domain} ' if sub_domain else ''}needs."

    def generate_domain_response(self, domain: str, responses: dict, sub_domain: str = None, original_prompt: str = None) -> str:
        """Generate a personalized response based on domain, sub-domain and collected information using LLM."""
        try:
            # Create a context-aware prompt for the LLM
            responses_str = json.dumps(responses, indent=2)
            
            llm_prompt = f"""
            Based on the following context, generate at least 15 highly personalized and realistic follow-up questions that are:
            - Selfish (focused on the user's needs, preferences, and goals)
            - Deeply related to the provided domain and sub-domain
            - Designed to gather more specific, relevant, or insightful information from the user
            - Written in a friendly, helpful tone

            Original Request: "{original_prompt}"
            Domain: {domain}
            Sub-domain: {sub_domain if sub_domain else "general"}

            Information already collected:
            {responses_str}

            The questions should:
            1. Respect what the user has already shared
            2. Avoid asking the same thing again
            3. Help us understand the user's preferences, needs, habits, or situations better
            4. Be diverse in angle (emotions, behavior, expectations, experiences, etc.)

            Return the output as a numbered list of questions. Avoid technical jargon unless the context clearly demands it.
            """

            
            llm = genai.GenerativeModel('gemini-2.0-flash')
            response = llm.generate_content(llm_prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating response with LLM: {e}")
            
            # Fallback response if LLM fails
            domain_responses = {
                "transportation": "I've updated your transportation preferences. I can help you with your travel plans based on the information you provided.",
                "retail": "I've updated your shopping preferences. I'll keep these in mind for future shopping recommendations.",
                "healthcare": "I've updated your healthcare preferences. I'll remember these for future healthcare appointments.",
                "legal": "I've updated your legal service preferences. I'll note these for future legal consultations.",
                "hospitality": "I've updated your dining preferences. I'll remember these for future restaurant recommendations.",
                "education": "I've updated your education preferences. I'll keep these in mind for future learning opportunities.",
                "entertainment": "I've updated your entertainment preferences. I'll use these for future entertainment suggestions.",
                "real_estate": "I've updated your property preferences. I'll keep these in mind for future real estate inquiries.",
                "fitness": "I've updated your fitness preferences. I'll remember these for future fitness recommendations.",
                "travel_planning": "I've updated your travel preferences. I'll use these for future travel planning assistance."
            }
            
            sub_domain_text = f" for {sub_domain.replace('_', ' ')}" if sub_domain else ""
            return domain_responses.get(domain, f"I've updated your {domain} preferences{sub_domain_text}.")


if __name__ == "__main__":
    agent = AIAgent()
    agent.start()