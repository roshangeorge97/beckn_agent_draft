from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict
from enum import Enum

class FollowUp(BaseModel):
    id: str
    question: str
    response: str
    timestamp: str

class ContactMethod(str, Enum):
    EMAIL = "email"
    PHONE = "phone"

class Location(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    preferred_timezone: Optional[str] = None

class Route(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    frequency: Optional[str] = None
    preferred_class: Optional[str] = None

# New base class for domain models
class DomainBase(BaseModel):
    follow_up_data: Dict[str, str] = Field(default_factory=dict)

class Transportation(DomainBase):
    preferred_mode: Optional[str] = None
    frequent_routes: Optional[str] = None
    seat_preference: Optional[str] = None
    trip_type: Optional[str] = None
    budget_range: Optional[str] = None
    preferred_time: Optional[str] = None
    loyalty_programs: List[str] = Field(default_factory=list)
    baggage_needs: Optional[str] = None
    meal_preference: Optional[str] = None
    accessibility_needs: Optional[str] = None
    travel_frequency: Optional[str] = None
    preferred_airlines: List[str] = Field(default_factory=list)
    stopover_preference: Optional[str] = None
    booking_platform: Optional[str] = None
    travel_insurance: Optional[str] = None
    eco_friendly_options: Optional[str] = None
    travel_document_type: Optional[str] = None
    airport_proximity_importance: Optional[str] = None
    emergency_travel_support: Optional[bool] = None
    follow_ups: List[FollowUp] = []

class Retail(DomainBase):
    preferred_categories: List[str] = Field(default_factory=list)
    brand_preferences: List[str] = Field(default_factory=list)
    budget_range: Optional[str] = None
    delivery_preference: Optional[str] = None
    shopping_frequency: Optional[str] = None
    store_type: Optional[str] = None
    loyalty_memberships: List[str] = Field(default_factory=list)
    product_quality_preference: Optional[str] = None
    return_policy_importance: Optional[str] = None
    seasonal_shopping: Optional[str] = None
    online_vs_instore: Optional[str] = None
    gift_shopping_frequency: Optional[str] = None
    discount_preference: Optional[str] = None
    eco_friendly_products: Optional[str] = None
    customer_service_priority: Optional[str] = None
    product_reviews_reliance: Optional[str] = None
    preferred_payment_methods: List[str] = Field(default_factory=list)
    wishlist_items: List[str] = Field(default_factory=list)
    follow_ups: List[FollowUp] = []

class Healthcare(DomainBase):
    preferred_specialties: List[str] = Field(default_factory=list)
    insurance_provider: Optional[str] = None
    appointment_type: Optional[str] = None
    preferred_time: Optional[str] = None
    location_preference: Optional[str] = None
    telehealth_preference: Optional[str] = None
    doctor_gender_preference: Optional[str] = None
    appointment_frequency: Optional[str] = None
    health_goals: Optional[str] = None
    preferred_clinic_type: Optional[str] = None
    prescription_management: Optional[str] = None
    wellness_services: List[str] = Field(default_factory=list)
    emergency_contact: Optional[str] = None
    medical_history_sharing: Optional[str] = None
    appointment_reminders: Optional[str] = None
    specialist_referral: Optional[str] = None
    chronic_conditions: List[str] = Field(default_factory=list)
    vaccination_records: List[str] = Field(default_factory=list)
    primary_care_physician: Optional[str] = None
    follow_ups: List[FollowUp] = []

class Legal(DomainBase):
    preferred_services: List[str] = Field(default_factory=list)
    urgency_level: Optional[str] = None
    budget_range: Optional[str] = None
    consultation_mode: Optional[str] = None
    lawyer_specialization: Optional[str] = None
    case_type: Optional[str] = None
    previous_legal_experience: Optional[str] = None
    document_preparation_needs: Optional[str] = None
    court_representation: Optional[str] = None
    contract_review_frequency: Optional[str] = None
    preferred_firm_size: Optional[str] = None
    confidentiality_level: Optional[str] = None
    consultation_duration: Optional[str] = None
    legal_aid_eligibility: Optional[str] = None
    dispute_resolution_preference: Optional[str] = None
    legal_tech_usage: Optional[str] = None
    legal_document_storage: Optional[str] = None
    previous_case_outcome: Optional[str] = None
    follow_ups: List[FollowUp] = []

class Hospitality(DomainBase):
    preferred_cuisine: List[str] = Field(default_factory=list)
    dining_style: Optional[str] = None
    budget_range: Optional[str] = None
    reservation_time: Optional[str] = None
    dietary_restrictions: List[str] = Field(default_factory=list)
    ambiance_preference: Optional[str] = None
    group_size: Optional[str] = None
    occasion_type: Optional[str] = None
    seating_preference: Optional[str] = None
    takeout_frequency: Optional[str] = None
    beverage_preference: Optional[str] = None
    loyalty_programs: List[str] = Field(default_factory=list)
    chef_specials_interest: Optional[str] = None
    dining_location: Optional[str] = None
    tipping_habits: Optional[str] = None
    food_allergies: List[str] = Field(default_factory=list)
    preferred_reservation_platform: Optional[str] = None
    pet_friendly_restaurants: Optional[bool] = None
    follow_ups: List[FollowUp] = []

class Education(DomainBase):
    learning_mode: Optional[str] = None
    subjects: List[str] = Field(default_factory=list)
    level: Optional[str] = None
    schedule_preference: Optional[str] = None
    learning_goals: Optional[str] = None
    course_format: Optional[str] = None
    instructor_preference: Optional[str] = None
    certification_needs: Optional[str] = None
    study_environment: Optional[str] = None
    group_vs_individual: Optional[str] = None
    technology_usage: Optional[str] = None
    learning_pace: Optional[str] = None
    resource_access: Optional[str] = None
    extracurricular_interests: List[str] = Field(default_factory=list)
    mentorship_preference: Optional[str] = None
    assessment_style: Optional[str] = None
    academic_background: Optional[str] = None
    preferred_languages: List[str] = Field(default_factory=list)
    follow_ups: List[FollowUp] = []

class Entertainment(DomainBase):
    genres: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    event_type: Optional[str] = None
    budget_range: Optional[str] = None
    viewing_frequency: Optional[str] = None
    live_event_preference: Optional[str] = None
    streaming_service: List[str] = Field(default_factory=list)
    content_rating: Optional[str] = None
    social_viewing: Optional[str] = None
    device_preference: Optional[str] = None
    content_discovery: Optional[str] = None
    merchandise_interest: Optional[str] = None
    event_location: Optional[str] = None
    subscription_model: Optional[str] = None
    content_language: Optional[str] = None
    interactive_content: Optional[str] = None
    preferred_actors_or_creators: List[str] = Field(default_factory=list)
    rewatch_habits: Optional[bool] = None
    follow_ups: List[FollowUp] = []

class RealEstate(DomainBase):
    property_type: Optional[str] = None
    budget_range: Optional[str] = None
    location_preferences: List[str] = Field(default_factory=list)
    amenities: List[str] = Field(default_factory=list)
    purchase_type: Optional[str] = None
    property_size: Optional[str] = None
    commute_preference: Optional[str] = None
    neighborhood_type: Optional[str] = None
    financing_preference: Optional[str] = None
    inspection_priority: Optional[str] = None
    pet_friendly: Optional[str] = None
    renovation_needs: Optional[str] = None
    outdoor_space: Optional[str] = None
    security_features: Optional[str] = None
    smart_home_features: Optional[str] = None
    community_amenities: List[str] = Field(default_factory=list)
    resale_value_importance: Optional[str] = None
    future_expansion_plan: Optional[bool] = None
    follow_ups: List[FollowUp] = []

class Fitness(DomainBase):
    activity_type: List[str] = Field(default_factory=list)
    schedule_preference: Optional[str] = None
    trainer_preference: Optional[str] = None
    budget_range: Optional[str] = None
    fitness_goals: Optional[str] = None
    workout_location: Optional[str] = None
    equipment_access: Optional[str] = None
    group_class_interest: Optional[str] = None
    fitness_level: Optional[str] = None
    dietary_integration: Optional[str] = None
    tracking_tools: List[str] = Field(default_factory=list)
    recovery_methods: List[str] = Field(default_factory=list)
    workout_frequency: Optional[str] = None
    outdoor_activities: Optional[str] = None
    wellness_integration: Optional[str] = None
    injury_history: Optional[str] = None
    workout_duration: Optional[str] = None
    preferred_music_genres: List[str] = Field(default_factory=list)
    follow_ups: List[FollowUp] = []

class TravelPlanning(DomainBase):
    travel_type: Optional[str] = None
    destinations: List[str] = Field(default_factory=list)
    budget_range: Optional[str] = None
    travel_companions: Optional[str] = None
    trip_duration: Optional[str] = None
    accommodation_type: Optional[str] = None
    activity_preferences: List[str] = Field(default_factory=list)
    travel_season: Optional[str] = None
    cultural_interests: Optional[str] = None
    adventure_level: Optional[str] = None
    transportation_preference: Optional[str] = None
    itinerary_style: Optional[str] = None
    travel_insurance: Optional[str] = None
    visa_requirements: Optional[str] = None
    local_guide_preference: Optional[str] = None
    sustainable_travel: Optional[str] = None
    packing_preferences: Optional[str] = None
    language_support_needs: Optional[str] = None
    follow_ups: List[FollowUp] = []

class GeneralInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Location = Field(default_factory=Location)
    preferred_contact_method: Optional[ContactMethod] = None

class Interaction(BaseModel):
    domain: str
    timestamp: str
    query: str
    response: Dict

class UserProfile(BaseModel):
    user_id: str
    general_info: GeneralInfo = Field(default_factory=GeneralInfo)
    domains: Dict[str, BaseModel] = Field(default_factory=lambda: {
        "transportation": Transportation(),
        "retail": Retail(),
        "healthcare": Healthcare(),
        "legal": Legal(),
        "hospitality": Hospitality(),
        "education": Education(),
        "entertainment": Entertainment(),
        "real_estate": RealEstate(),
        "fitness": Fitness(),
        "travel_planning": TravelPlanning()
    })
    interaction_history: List[Interaction] = Field(default_factory=list)