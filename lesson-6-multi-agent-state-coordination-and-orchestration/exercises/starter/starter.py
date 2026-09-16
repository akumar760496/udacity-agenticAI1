"""
Pasta Factory Exercise - Starter Code
====================================

In this exercise, you'll extend the Italian Pasta Factory multi-agent system 
to handle more complex scenarios with shared state coordination and proper
multi-agent orchestration patterns.

You'll need to:
1. Implement the missing production and custom recipe tools
2. Create the CustomPastaDesignerAgent
3. Build the proper Orchestrator using ToolCallingAgent
4. Add coordination tools that route requests between specialized agents

This demonstrates extending multi-agent systems with new capabilities while
maintaining proper orchestration patterns.
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import random
from dataclasses import dataclass, field, asdict

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

# Load your OpenAI API key
import os
import dotenv
dotenv.load_dotenv(dotenv_path="../.env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key="voc-153329278615876653870466aa4d89c977bb1.17260821", #os.getenv("UDACITY_OPENAI_API_KEY"),
    api_base="https://openai.vocareum.com/v1",
)

# Pasta Factory State Management

@dataclass
class PastaOrder:
    order_id: str
    pasta_shape: str
    quantity: float  # in kg
    status: str = "pending"  # pending, queued, completed, cancelled
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 1  # 1 = normal, 2 = rush, 3 = emergency
    customer_notes: str = ""
    estimated_delivery_date: str = ""

@dataclass
class FactoryState:
    inventory: Dict[str, float] = field(default_factory=lambda: {
        "flour": 10.0,  # kg
        "water": 5.0,   # liters
        "eggs": 24,     # count
        "semolina": 8.0 # kg
    })
    production_queue: List[PastaOrder] = field(default_factory=list)
    pasta_recipes: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "spaghetti": {"flour": 0.2, "water": 0.1},
        "fettuccine": {"flour": 0.25, "water": 0.1},
        "penne": {"flour": 0.2, "water": 0.1},
        "ravioli": {"flour": 0.3, "water": 0.1, "eggs": 2},
        "lasagna": {"flour": 0.3, "water": 0.15, "eggs": 3}
    })
    custom_recipes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    order_counter: int = 0
    known_pasta_shapes: List[str] = field(default_factory=lambda: [
        "spaghetti", "fettuccine", "penne", "ravioli", "lasagna"
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inventory": self.inventory,
            "production_queue": [asdict(order) for order in self.production_queue],
            "pasta_recipes": self.pasta_recipes,
            "custom_recipes": self.custom_recipes
        }
    
    def update_known_pasta_shapes(self):
        """Update the list of known pasta shapes based on recipes."""
        self.known_pasta_shapes = list(self.pasta_recipes.keys()) + list(self.custom_recipes.keys())

# Initialize the shared factory state
factory_state = FactoryState()

# ======= Agent Tools =======

@tool
def check_pasta_recipe(pasta_shape: str) -> Dict[str, float]:
    """
    Check what ingredients are needed for a specific pasta shape.
    Returns a dictionary of ingredients and amounts needed per kg of pasta.
    
    Args:
        pasta_shape: Name of the pasta shape to check recipes for
    """
    if pasta_shape in factory_state.pasta_recipes:
        return factory_state.pasta_recipes[pasta_shape]
    elif pasta_shape in factory_state.custom_recipes:
        return factory_state.custom_recipes[pasta_shape]
    return {}

@tool
def check_inventory() -> Dict[str, float]:
    """Check current inventory levels of all ingredients."""
    return factory_state.inventory

@tool
def generate_order_id() -> str:
    """Generate a unique order ID."""
    factory_state.order_counter += 1
    return f"ORD-{factory_state.order_counter:04d}"

@tool
def list_available_pasta_shapes() -> List[str]:
    """List all available pasta shapes that can be ordered."""
    return factory_state.known_pasta_shapes

@tool
def update_inventory(ingredient: str, amount: float) -> Dict[str, Any]:
    """
    Update the inventory amount for a specific ingredient.
    
    Args:
        ingredient: Name of the ingredient
        amount: New amount (will replace current amount)
        
    Returns:
        Status of the inventory update
    """
    if ingredient not in factory_state.inventory:
        return {
            "success": False,
            "message": f"Unknown ingredient: {ingredient}. Cannot update inventory."
        }
    
    old_amount = factory_state.inventory[ingredient]
    factory_state.inventory[ingredient] = amount
    
    return {
        "success": True,
        "message": f"Inventory updated: {ingredient} from {old_amount} to {amount}.",
        "ingredient": ingredient,
        "old_amount": old_amount,
        "new_amount": amount
    }

@tool
def check_production_capacity(days_ahead: int = 7) -> Dict[str, Any]:
    """
    Check the current production capacity and queue for the next X days.
    Returns information about queue size and estimated completion times.
    
    Args:
        days_ahead: Number of days ahead to project capacity for
    """
    queue_size = len(factory_state.production_queue)
    
    # Calculate the total production volume (in kg)
    total_volume = sum(order.quantity for order in factory_state.production_queue)
    
    # Simple capacity estimation: assume we can produce 10kg per day
    daily_capacity = 10.0  # kg per day
    days_to_complete = max(1, total_volume / daily_capacity)
    
    # Consider priority orders
    priority_orders = [o for o in factory_state.production_queue if o.priority > 1]
    priority_volume = sum(order.quantity for order in priority_orders)
    
    return {
        "queue_size": queue_size,
        "total_volume_kg": total_volume,
        "days_to_complete_current_queue": days_to_complete,
        "daily_capacity_kg": daily_capacity,
        "priority_orders": len(priority_orders),
        "priority_volume_kg": priority_volume
    }

# ======= Implemented TODO Tools =======

@tool
def add_to_production_queue(
    order_id: str,
    pasta_shape: str,
    quantity: float,
    priority: int = 1,
    customer_notes: str = ""
) -> Dict[str, Any]:
    """
    Add an order to the production queue.
    
    Args:
        order_id: Unique order identifier
        pasta_shape: Type of pasta to produce
        quantity: Amount in kg
        priority: Order priority (1=normal, 2=rush, 3=emergency)
        customer_notes: Additional notes from customer
        
    Returns:
        Status of the queuing operation with estimated delivery date
    """
    recipe = check_pasta_recipe(pasta_shape)
    if not recipe:
        return {"success": False, "message": f"Recipe for pasta shape '{pasta_shape}' not found."}
    
    # Check ingredient inventory requirements
    required_ingredients = {ing: amt * quantity for ing, amt in recipe.items()}
    for ing, req_amt in required_ingredients.items():
        if factory_state.inventory.get(ing, 0.0) < req_amt:
            return {
                "success": False, 
                "message": f"Insufficient inventory for {ing}. Required: {req_amt}, Available: {factory_state.inventory.get(ing, 0.0)}"
            }
            
    # Deduct ingredients from inventory
    for ing, req_amt in required_ingredients.items():
        factory_state.inventory[ing] -= req_amt
        
    # Calculate delivery date based on priority
    days_offset = max(1, int(quantity / 10.0))
    if priority == 2:
        days_offset = max(1, days_offset // 2)
    elif priority == 3:
        days_offset = 0  # Same day
        
    delivery_date = (datetime.now() + timedelta(days=days_offset)).strftime("%Y-%m-%d")
    
    order = PastaOrder(
        order_id=order_id,
        pasta_shape=pasta_shape,
        quantity=quantity,
        status="queued",
        priority=priority,
        customer_notes=customer_notes,
        estimated_delivery_date=delivery_date
    )
    
    factory_state.production_queue.append(order)
    
    return {
        "success": True,
        "order_id": order_id,
        "estimated_delivery_date": delivery_date,
        "message": f"Successfully added order {order_id} for {quantity}kg of {pasta_shape} to production queue."
    }

@tool
def create_custom_pasta_recipe(
    pasta_name: str,
    ingredients: Dict[str, float]
) -> Dict[str, Any]:
    """
    Create a custom pasta recipe with specific ingredient ratios.
    
    Args:
        pasta_name: Name of the custom pasta
        ingredients: Dictionary mapping ingredient names to amounts needed per kg
        
    Returns:
        Status of the recipe creation
    """
    for ing in ingredients.keys():
        if ing not in factory_state.inventory:
            return {"success": False, "message": f"Unknown inventory ingredient: {ing}"}
            
    if pasta_name in factory_state.pasta_recipes or pasta_name in factory_state.custom_recipes:
        return {"success": False, "message": f"Recipe for '{pasta_name}' already exists."}
        
    factory_state.custom_recipes[pasta_name] = ingredients
    factory_state.update_known_pasta_shapes()
    
    return {
        "success": True,
        "pasta_name": pasta_name,
        "ingredients": ingredients,
        "message": f"Custom pasta recipe '{pasta_name}' created successfully."
    }

@tool
def prioritize_order(order_id: str, new_priority: int) -> Dict[str, Any]:
    """
    Change the priority of an existing order in the queue.
    
    Args:
        order_id: ID of the order to update
        new_priority: New priority level (1=normal, 2=rush, 3=emergency)
        
    Returns:
        Status of the priority change
    """
    if new_priority not in [1, 2, 3]:
        return {"success": False, "message": "Invalid priority level. Must be 1, 2, or 3."}
        
    for order in factory_state.production_queue:
        if order.order_id == order_id:
            order.priority = new_priority
            days_offset = 0 if new_priority == 3 else (1 if new_priority == 2 else 3)
            order.estimated_delivery_date = (datetime.now() + timedelta(days=days_offset)).strftime("%Y-%m-%d")
            return {
                "success": True,
                "order_id": order_id,
                "new_priority": new_priority,
                - "estimated_delivery_date": order.estimated_delivery_date,
                "message": f"Order {order_id} priority updated to {new_priority}."
            }
            
    return {"success": False, "message": f"Order ID {order_id} not found in production queue."}

# ======= Agents =======

class OrderProcessorAgent(ToolCallingAgent):
    """Agent responsible for processing customer order requests."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_pasta_recipe, generate_order_id, list_available_pasta_shapes],
            model=model,
            name="order_processor",
            description="Agent responsible for processing customer orders. Parses requests, identifies pasta shapes and quantities."
        )

class InventoryManagerAgent(ToolCallingAgent):
    """Agent responsible for managing ingredient inventory."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_inventory, check_pasta_recipe, update_inventory],
            model=model,
            name="inventory_manager",
            description="Agent responsible for tracking and managing ingredient inventory."
        )

class ProductionManagerAgent(ToolCallingAgent):
    """Agent responsible for managing the production queue."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_production_capacity, add_to_production_queue, prioritize_order],
            model=model,
            name="production_manager",
            description="Agent responsible for managing production scheduling, queue management, and order prioritization."
        )

class CustomPastaDesignerAgent(ToolCallingAgent):
    """Agent responsible for designing custom pasta recipes."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_inventory, create_custom_pasta_recipe],
            model=model,
            name="pasta_designer",
            description="Specialist agent responsible for creating and validating custom pasta recipes based on available ingredients.",
        )

# ======= Orchestrator =======

class Orchestrator(ToolCallingAgent):
    """Orchestrator that coordinates workflow between specialized agents."""
    
    def __init__(self, model):
        self.model = model
        
        self.order_processor = OrderProcessorAgent(model)
        self.inventory_manager = InventoryManagerAgent(model)
        self.production_manager = ProductionManagerAgent(model)
        self.pasta_designer = CustomPastaDesignerAgent(model)

        @tool
        def process_order_info(customer_request: str) -> str:
            """Process customer order information to extract details.
            
            Args:
                customer_request: The customer's order request
                
            Returns:
                Processed order information with pasta shape and quantity
            """
            return self.order_processor.run(customer_request)

        @tool
        def manage_inventory(order_details: str) -> str:
            """Check and manage inventory for an order.
            
            Args:
                order_details: Details about the order including pasta shape and quantity
                
            Returns:
                Inventory management result
            """
            return self.inventory_manager.run(order_details)

        @tool
        def schedule_production(order_info: str, priority: int = 1) -> str:
            """Schedule production for an order.
            
            Args:
                order_info: Information about the order to schedule
                priority: Order priority (1=normal, 2=rush, 3=emergency)
                
            Returns:
                Production scheduling result with delivery date
            """
            return self.production_manager.run(f"Schedule order details: {order_info} with priority {priority}")

        @tool
        def design_custom_pasta(customer_request: str) -> str:
            """Design a custom pasta recipe based on customer requirements.
            
            Args:
                customer_request: Customer's custom pasta request
                
            Returns:
                Custom pasta design result
            """
            return self.pasta_designer.run(customer_request)

        super().__init__(
            tools=[process_order_info, manage_inventory, schedule_production, design_custom_pasta],
            model=model,
            name="orchestrator",
            description="""
            Orchestrator for the pasta factory system that coordinates workflow 
            between specialized agents for order processing, inventory management, 
            production scheduling, and custom pasta design.
            """,
        )
        
    def process_order(self, customer_request: str) -> str:
        """
        Process a customer order through coordinated agent workflow.
        """
        prompt = f"""
        Handle the following customer request thoroughly:
        "{customer_request}"
        
        Determine if this is a custom pasta recipe request or a standard order, 
        check inventory levels, generate an order ID, and queue it into production 
        with appropriate priority (1 for normal, 2 for rush/tomorrow, 3 for emergency). 
        Provide a final clear summary response to the customer.
        """
        return self.run(prompt)

# ======= Main Demo =======

def run_demo():
    """Run a demonstration of the pasta factory system."""
    orchestrator = Orchestrator(model)
    
    print("Welcome to the Pasta Factory Multi-Agent System!")
    print("Initial Factory State:", json.dumps(factory_state.to_dict(), indent=2))
    
    orders = [
        "I'd like to order 2kg of spaghetti please. When can I get it?",
        "I need a custom pasta called 'semolina_special' with 0.3kg semolina and 0.1kg water per kg. Can you make that?",
        "Rush order! We need 5kg of fettuccine for a catering event tomorrow!",
    ]
    
    for i, order in enumerate(orders):
        print(f"\n--- Processing Order {i+1} ---")
        print(f"Customer: {order}")
        
        response = orchestrator.process_order(order)
        print(f"Factory: {response}")
        
    print("\n--- Final Factory State ---")
    print(json.dumps(factory_state.to_dict(), indent=2))
    print("\nDemo complete! This demonstrates multi-agent coordination with shared state management.")

if __name__ == "__main__":
    run_demo()