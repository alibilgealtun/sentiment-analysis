"""
Data Generator Module

This module generates synthetic technician feedback data for sentiment analysis.
The generated data includes realistic technical service scenarios with various
sentiment categories.

Author: Sentiment Analysis Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Optional, Dict, List, Tuple


# Feedback templates organized by sentiment and category
FEEDBACK_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "positive": {
        "equipment": [
            "The new diagnostic tool has significantly improved our repair efficiency",
            "Equipment performance has been excellent this quarter",
            "The upgraded machinery is working flawlessly",
            "Our new testing equipment is incredibly accurate and reliable",
            "The calibration tools are top-notch and easy to use",
            "Really impressed with the new multimeter's precision",
            "The thermal imaging camera has been a game changer for diagnostics",
            "New oscilloscope is fantastic for circuit troubleshooting",
            "The pneumatic tools are much more powerful and efficient now",
            "Updated hydraulic systems are performing beyond expectations",
            "The laser alignment tool has made our work much more precise",
            "Electrical testing equipment upgrade was worth every penny",
            "The new welding machine produces consistently clean welds",
            "Our diagnostic software accurately identifies issues quickly",
            "The portable generator is reliable and starts every time",
        ],
        "service": [
            "Customer feedback has been overwhelmingly positive lately",
            "Service delivery times have improved dramatically",
            "Client satisfaction scores are at an all-time high",
            "We've received excellent reviews for our maintenance work",
            "The new service protocol has streamlined our operations",
            "Response time to service calls has improved significantly",
            "Customers are very happy with our preventive maintenance program",
            "Our warranty service has received great feedback",
            "Installation services are running smoothly and efficiently",
            "Field service team has been doing outstanding work",
            "Emergency repair response has been exceptional",
            "Customer retention rate has increased this quarter",
            "Service quality metrics are exceeding targets",
            "Our technical support team is getting praise from clients",
            "After-sales service satisfaction is at peak levels",
        ],
        "training": [
            "The recent safety training was very comprehensive and useful",
            "New onboarding program has been extremely effective",
            "Technical certification courses have improved our skills",
            "The hands-on workshop was incredibly valuable",
            "Cross-training program has enhanced team versatility",
            "Online learning modules are well-designed and accessible",
            "Manufacturer training sessions provided excellent insights",
            "The mentorship program has accelerated skill development",
            "Safety drills are well-organized and informative",
            "Technical documentation training was very helpful",
            "The advanced troubleshooting course was outstanding",
            "Leadership development program exceeded expectations",
            "Quality control training has improved our standards",
            "The certification prep course was comprehensive",
            "Continuous learning opportunities are appreciated",
        ],
        "safety": [
            "New safety protocols have significantly reduced incidents",
            "PPE quality has improved substantially",
            "Safety culture in our team is exemplary",
            "The updated safety equipment is much more comfortable",
            "Emergency response procedures are well-established",
            "Safety briefings are thorough and informative",
            "Hazard identification training has been valuable",
            "The new ventilation system has improved air quality",
            "Safety inspections are conducted regularly and thoroughly",
            "First aid kits are well-stocked and accessible",
            "Fire safety measures are comprehensive and effective",
            "Lockout/tagout procedures are clearly documented",
            "Safety signage improvements have increased awareness",
            "Personal monitoring devices are accurate and reliable",
            "The ergonomic improvements have reduced strain injuries",
        ],
        "workload": [
            "Work scheduling has become much more manageable",
            "The new shift system allows for better work-life balance",
            "Task distribution is fair and well-organized",
            "Workload management tools have improved efficiency",
            "Overtime has decreased while productivity increased",
            "Project timelines are realistic and achievable",
            "Resource allocation has been optimized effectively",
            "Team collaboration has made heavy workloads manageable",
            "The priority system helps focus on critical tasks",
            "Flexible scheduling options are greatly appreciated",
            "Automated systems have reduced manual workload",
            "Task tracking software has improved organization",
            "Break times are respected and adequate",
            "Seasonal workload variations are well-planned",
            "Support staff availability has improved workflow",
        ],
        "management": [
            "Leadership team is supportive and responsive",
            "Management communication has improved significantly",
            "Supervisors provide constructive feedback regularly",
            "Decision-making process is transparent and inclusive",
            "Management recognizes and rewards good performance",
            "Open-door policy encourages honest communication",
            "Team meetings are productive and well-organized",
            "Management invests in employee development",
            "Performance reviews are fair and comprehensive",
            "Leadership handles conflicts professionally",
            "Strategic direction is clearly communicated",
            "Management listens to technician suggestions",
            "Resource requests are processed efficiently",
            "Career advancement opportunities are available",
            "Management maintains high ethical standards",
        ],
        "tools": [
            "The new tool organization system saves valuable time",
            "Tool quality has improved with recent upgrades",
            "Specialized tools are now readily available",
            "Tool maintenance schedule keeps everything in top condition",
            "The digital torque wrench is extremely precise",
            "Power tools are well-maintained and reliable",
            "The tool inventory system is efficient and accurate",
            "Hand tools have ergonomic grips that reduce fatigue",
            "Diagnostic tools interface seamlessly with our systems",
            "Tool replacement process is quick and hassle-free",
            "The mobile tool cart is well-designed and practical",
            "Precision measuring instruments are accurately calibrated",
            "Cordless tools have excellent battery life",
            "Tool storage solutions are well-organized",
            "Safety features on power tools are excellent",
        ],
        "communication": [
            "Team communication channels are effective and clear",
            "The new communication app has improved coordination",
            "Information sharing between shifts is seamless",
            "Technical documentation is well-written and accessible",
            "Daily briefings keep everyone informed",
            "Communication with other departments has improved",
            "The ticketing system provides clear job details",
            "Customer communication protocols work well",
            "Emergency communication procedures are reliable",
            "Feedback channels are open and responsive",
            "Digital work orders are clear and complete",
            "Team meetings foster good communication",
            "The knowledge base is comprehensive and helpful",
            "Inter-department collaboration is excellent",
            "Remote communication tools work reliably",
        ],
    },
    "negative": {
        "equipment": [
            "Equipment keeps breaking down, we need better maintenance schedules",
            "The old machinery is constantly malfunctioning",
            "Diagnostic equipment gives inconsistent readings",
            "We desperately need upgraded testing tools",
            "Equipment failures are causing significant delays",
            "The aging fleet requires immediate attention",
            "Calibration issues are affecting measurement accuracy",
            "Replacement parts for equipment are hard to source",
            "The outdated software is incompatible with new systems",
            "Equipment downtime is severely impacting productivity",
            "Maintenance costs for old equipment are excessive",
            "Critical equipment lacks proper backup systems",
            "The faulty sensors are causing misdiagnoses",
            "Power tools are worn out and need replacement",
            "Safety features on equipment are not functioning properly",
        ],
        "service": [
            "Service backlogs are causing customer complaints",
            "Response times to service requests are too long",
            "Quality of repairs has declined recently",
            "Customer satisfaction scores have dropped significantly",
            "Service scheduling system is inefficient and confusing",
            "We're unable to meet service level agreements",
            "Parts availability issues delay service completion",
            "Field service coordination needs improvement",
            "Warranty claim processing is too slow",
            "Service documentation is incomplete and outdated",
            "Customer callback rate is unacceptably high",
            "Emergency service response is inadequate",
            "Service pricing structure is confusing customers",
            "Technical support quality has deteriorated",
            "Installation services are facing repeated issues",
        ],
        "training": [
            "Training programs are outdated and irrelevant",
            "Insufficient training for new equipment",
            "Safety training is not conducted frequently enough",
            "Technical certification opportunities are limited",
            "Training materials are poorly designed",
            "No time allocated for skill development",
            "New hires receive inadequate onboarding",
            "Advanced training courses are not available",
            "Training budget has been cut significantly",
            "Hands-on practice opportunities are lacking",
            "Training schedules conflict with work demands",
            "Instructor quality is inconsistent",
            "No access to manufacturer training programs",
            "Cross-training between departments is nonexistent",
            "Online training platform is difficult to use",
        ],
        "safety": [
            "Safety equipment is outdated and uncomfortable",
            "Safety protocols are not being followed consistently",
            "Near-miss incidents are not being reported properly",
            "PPE supply is frequently insufficient",
            "Safety training needs to be more comprehensive",
            "Hazardous conditions are not addressed promptly",
            "Emergency exits are sometimes blocked",
            "Safety inspections are too infrequent",
            "First aid supplies are often depleted",
            "Ventilation in work areas is inadequate",
            "Chemical handling procedures need updating",
            "Safety signage is unclear or missing",
            "Noise levels exceed safe thresholds",
            "Ergonomic concerns are being ignored",
            "Fall protection equipment needs replacement",
        ],
        "workload": [
            "Workload is overwhelming and unsustainable",
            "Unrealistic deadlines are causing burnout",
            "Staffing levels are insufficient for demand",
            "Overtime requirements are excessive",
            "Task distribution is unfair and inconsistent",
            "No time for breaks during busy periods",
            "Work-life balance has deteriorated significantly",
            "Emergency calls are too frequent",
            "Administrative tasks take time from technical work",
            "Project scope keeps expanding without resources",
            "Peak season workload is unmanageable",
            "Multiple priorities make focusing impossible",
            "Backlog of work orders is growing",
            "Coverage for absences is inadequate",
            "Travel requirements are excessive",
        ],
        "management": [
            "Management is unresponsive to technician concerns",
            "Communication from leadership is poor",
            "Decisions are made without consulting the team",
            "Recognition for good work is nonexistent",
            "Supervisors lack technical knowledge",
            "Performance metrics are unfair and unrealistic",
            "Career advancement opportunities are unclear",
            "Management plays favorites with assignments",
            "Feedback from technicians is ignored",
            "Resource requests are consistently denied",
            "Policy changes are implemented without notice",
            "Conflict resolution by management is ineffective",
            "Strategic direction is constantly changing",
            "Management accountability is lacking",
            "Team morale is at an all-time low",
        ],
        "tools": [
            "Tools are frequently missing or broken",
            "Tool quality is substandard for professional work",
            "No budget for necessary tool upgrades",
            "Tool organization system is chaotic",
            "Specialized tools are not available when needed",
            "Tool maintenance is neglected",
            "Shared tools are often in poor condition",
            "Tool inventory is inaccurate",
            "Outdated tools slow down our work",
            "Replacement tools take too long to arrive",
            "Hand tools lack proper storage",
            "Diagnostic tools are outdated",
            "Power tool batteries don't hold charge",
            "Precision tools are out of calibration",
            "Tool checkout system doesn't work",
        ],
        "communication": [
            "Communication between shifts is poor",
            "Important information is not shared timely",
            "Technical documentation is outdated or missing",
            "Work orders lack necessary details",
            "No clear escalation procedures",
            "Miscommunication causes repeated work",
            "Department silos prevent collaboration",
            "Customer communication is inconsistent",
            "Meeting frequency is excessive and unproductive",
            "Email overload makes important messages hard to find",
            "Communication tools are unreliable",
            "Feedback is not acted upon",
            "Language barriers affect team communication",
            "Remote teams are often out of the loop",
            "Critical updates are buried in long emails",
        ],
    },
    "neutral": {
        "equipment": [
            "Completed the scheduled maintenance as per the manual",
            "Equipment is functioning within normal parameters",
            "Routine calibration was performed today",
            "Standard equipment inspection completed",
            "Replaced worn parts during scheduled maintenance",
            "Equipment logs have been updated",
            "Normal wear observed during inspection",
            "Preventive maintenance checklist completed",
            "Equipment operating as expected",
            "Inventory of spare parts has been documented",
            "Standard operating procedures followed",
            "Equipment usage logged for the shift",
            "Routine cleaning and maintenance done",
            "No anomalies detected during inspection",
            "Equipment serial numbers verified",
        ],
        "service": [
            "Service call completed within standard timeframe",
            "Customer acknowledged receipt of service",
            "Standard service procedure was followed",
            "Service ticket has been closed",
            "Parts were installed according to specifications",
            "Documentation submitted for the service visit",
            "Customer signed off on completed work",
            "Follow-up appointment scheduled as requested",
            "Service report filed in the system",
            "Warranty information provided to customer",
            "Standard diagnostic procedure completed",
            "Service invoice generated and sent",
            "Customer preferences noted in file",
            "Routine service visit completed",
            "Next scheduled service date confirmed",
        ],
        "training": [
            "Attended the mandatory safety refresher course",
            "Training attendance logged in the system",
            "Completed required online module",
            "Certification renewal is due next month",
            "Training materials were distributed",
            "Standard training protocol followed",
            "New employee orientation completed",
            "Training hours documented for records",
            "Compliance training finished as scheduled",
            "Reviewed updated safety procedures",
            "Training room booked for next session",
            "Completed the annual review training",
            "Training certificate received",
            "Submitted training completion form",
            "Watched the safety video as required",
        ],
        "safety": [
            "Daily safety checklist completed",
            "PPE inventory count completed",
            "Safety meeting attendance recorded",
            "Monthly safety inspection conducted",
            "Fire extinguisher inspection logged",
            "Safety data sheets have been updated",
            "Emergency drill participation documented",
            "Safety compliance audit completed",
            "Reported standard safety observations",
            "Safety signage locations verified",
            "Reviewed lockout/tagout procedures",
            "Safety equipment locations confirmed",
            "Completed safety documentation",
            "Normal safety walk-through done",
            "Safety training roster updated",
        ],
        "workload": [
            "Standard number of work orders completed today",
            "Shift handover documentation completed",
            "Work schedule for next week received",
            "Task list for the day has been reviewed",
            "Workload is within normal range",
            "Time tracking submitted for the week",
            "Project status updated in the system",
            "Priority tasks identified for tomorrow",
            "Weekly work summary prepared",
            "Assigned tasks completed on schedule",
            "Work order backlog is unchanged",
            "Resource requirements documented",
            "Standard shift activities completed",
            "Task assignments acknowledged",
            "Work distribution reviewed",
        ],
        "management": [
            "Attended the regular team meeting",
            "Performance review is scheduled for next month",
            "Received the weekly update from management",
            "Submitted the monthly report as requested",
            "Team announcements were communicated",
            "Policy update acknowledged",
            "Participated in department planning session",
            "Management memo received and reviewed",
            "Standard check-in with supervisor completed",
            "Organizational update email reviewed",
            "Quarterly goals have been documented",
            "Standard approval process followed",
            "Meeting minutes distributed to team",
            "Resource allocation request submitted",
            "Budget update received from management",
        ],
        "tools": [
            "Tool inventory count completed",
            "Tools returned to storage after shift",
            "Standard tool maintenance performed",
            "Tool checkout logged in the system",
            "Daily tool inspection completed",
            "Tool organization per standard procedure",
            "Required tools prepared for tomorrow",
            "Tool calibration schedule reviewed",
            "Tool requisition form submitted",
            "Shared tools location updated",
            "Tool usage documented",
            "Standard tool kit contents verified",
            "Tool condition noted in log",
            "Tool replacement request submitted",
            "Equipment and tools secured for night",
        ],
        "communication": [
            "Daily briefing notes distributed",
            "Work order details communicated to team",
            "Customer contact information updated",
            "Shift change information passed on",
            "Standard communication channels used",
            "Meeting notes shared with team",
            "Email correspondence filed appropriately",
            "Technical update forwarded to colleagues",
            "Contact list has been updated",
            "Information request sent to vendor",
            "Standard report format followed",
            "Communication log maintained",
            "Team notification sent as required",
            "Documentation shared via standard channels",
            "Information acknowledged and filed",
        ],
    },
}


def generate_technician_feedback(
    n_samples: int = 500,
    sentiment_distribution: Optional[Dict[str, float]] = None,
    category_distribution: Optional[Dict[str, float]] = None,
    random_seed: int = 42,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate synthetic technician feedback data.
    
    Parameters
    ----------
    n_samples : int, default=500
        Number of feedback samples to generate.
    sentiment_distribution : dict, optional
        Distribution of sentiments. Keys should be 'positive', 'negative', 'neutral'.
        Values should sum to 1.0. Default is {'positive': 0.4, 'negative': 0.35, 'neutral': 0.25}.
    category_distribution : dict, optional
        Distribution of categories. Default is uniform distribution.
    random_seed : int, default=42
        Random seed for reproducibility.
    start_date : datetime, optional
        Start date for timestamp generation. Default is 1 year ago.
    end_date : datetime, optional
        End date for timestamp generation. Default is today.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, feedback_text, sentiment, category, timestamp
    
    Examples
    --------
    >>> df = generate_technician_feedback(n_samples=100)
    >>> print(df.head())
    
    >>> # Custom distribution
    >>> df = generate_technician_feedback(
    ...     n_samples=200,
    ...     sentiment_distribution={'positive': 0.5, 'negative': 0.3, 'neutral': 0.2}
    ... )
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Default sentiment distribution
    if sentiment_distribution is None:
        sentiment_distribution = {
            "positive": 0.40,
            "negative": 0.35,
            "neutral": 0.25,
        }
    
    # Default category distribution (uniform)
    categories = list(FEEDBACK_TEMPLATES["positive"].keys())
    if category_distribution is None:
        category_distribution = {cat: 1.0 / len(categories) for cat in categories}
    
    # Default date range (last 1 year)
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    # Validate distributions
    assert abs(sum(sentiment_distribution.values()) - 1.0) < 0.01, \
        "Sentiment distribution must sum to 1.0"
    assert abs(sum(category_distribution.values()) - 1.0) < 0.01, \
        "Category distribution must sum to 1.0"
    
    # Generate data
    data = []
    sentiments = list(sentiment_distribution.keys())
    sentiment_probs = list(sentiment_distribution.values())
    category_probs = [category_distribution[cat] for cat in categories]
    
    for i in range(n_samples):
        # Select sentiment and category based on distributions
        sentiment = np.random.choice(sentiments, p=sentiment_probs)
        category = np.random.choice(categories, p=category_probs)
        
        # Get feedback text
        feedback_options = FEEDBACK_TEMPLATES[sentiment][category]
        feedback_text = random.choice(feedback_options)
        
        # Add slight variations to make data more realistic
        feedback_text = _add_variations(feedback_text, random_seed + i)
        
        # Generate random timestamp
        days_diff = (end_date - start_date).days
        random_days = random.randint(0, days_diff)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        timestamp = start_date + timedelta(
            days=random_days,
            hours=random_hours,
            minutes=random_minutes
        )
        
        data.append({
            "id": i + 1,
            "feedback_text": feedback_text,
            "sentiment": sentiment,
            "category": category,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)
    
    return df


def _add_variations(text: str, seed: int) -> str:
    """
    Add slight variations to feedback text to make it more realistic.
    
    Parameters
    ----------
    text : str
        Original feedback text.
    seed : int
        Random seed for consistent variations.
    
    Returns
    -------
    str
        Modified feedback text with variations.
    """
    random.seed(seed)
    
    # Prefix variations
    prefixes = [
        "", "", "",  # Most often no prefix
        "I think ",
        "In my opinion, ",
        "Based on my experience, ",
        "Overall, ",
        "Generally speaking, ",
        "To be honest, ",
        "Honestly, ",
        "I believe ",
        "From what I've seen, ",
    ]
    
    # Suffix variations
    suffixes = [
        "", "", "", "",  # Most often no suffix
        ".",
        "!",
        " This needs attention.",
        " Looking forward to improvements.",
        " Thanks for listening.",
        " Please address this.",
        " Just my observation.",
        " Hope this helps.",
    ]
    
    # Apply variations with low probability
    result = text
    
    if random.random() < 0.15:  # 15% chance of prefix
        prefix = random.choice(prefixes)
        if prefix:
            # Lowercase the first letter of original text if adding prefix
            result = result[0].lower() + result[1:] if result else result
        result = prefix + result
    
    if random.random() < 0.10:  # 10% chance of suffix
        suffix = random.choice(suffixes)
        # Remove existing ending punctuation before adding suffix
        if result and result[-1] in '.!?':
            result = result[:-1]
        result = result + suffix
    
    return result


def get_sample_feedbacks(n_samples: int = 5) -> Dict[str, List[str]]:
    """
    Get sample feedback texts for each sentiment category.
    
    Parameters
    ----------
    n_samples : int, default=5
        Number of samples per sentiment.
    
    Returns
    -------
    dict
        Dictionary with sentiment as keys and list of sample texts as values.
    
    Examples
    --------
    >>> samples = get_sample_feedbacks(n_samples=3)
    >>> for sentiment, texts in samples.items():
    ...     print(f"{sentiment}: {len(texts)} samples")
    """
    samples = {}
    for sentiment in FEEDBACK_TEMPLATES:
        samples[sentiment] = []
        all_texts = []
        for category_texts in FEEDBACK_TEMPLATES[sentiment].values():
            all_texts.extend(category_texts)
        
        # Sample without replacement if possible
        n = min(n_samples, len(all_texts))
        samples[sentiment] = random.sample(all_texts, n)
    
    return samples


def save_dataset(
    df: pd.DataFrame,
    filepath: str = "data/technician_feedback.csv",
    index: bool = False
) -> None:
    """
    Save the generated dataset to a CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the feedback data.
    filepath : str, default="data/technician_feedback.csv"
        Path to save the CSV file.
    index : bool, default=False
        Whether to include the index in the CSV file.
    
    Examples
    --------
    >>> df = generate_technician_feedback(n_samples=500)
    >>> save_dataset(df, "data/technician_feedback.csv")
    """
    df.to_csv(filepath, index=index)
    print(f"Dataset saved to {filepath}")
    print(f"Total samples: {len(df)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts(normalize=True)}")


if __name__ == "__main__":
    # Generate and save the dataset
    print("Generating technician feedback dataset...")
    df = generate_technician_feedback(n_samples=500, random_seed=42)
    
    print("\nDataset Preview:")
    print(df.head(10))
    
    print("\nSentiment Distribution:")
    print(df["sentiment"].value_counts())
    
    print("\nCategory Distribution:")
    print(df["category"].value_counts())
    
    # Save to CSV
    save_dataset(df, "data/technician_feedback.csv")
