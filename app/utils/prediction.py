from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.who_standards import HeightForAge, WeightForAge
from app.models.child import Child


async def get_height_for_age_data(db: AsyncSession, age_years: int, gender: str):
    result = await db.execute(
        select(HeightForAge).filter(
            HeightForAge.year == age_years,
            HeightForAge.gender == gender
        )
    )
    return result.scalars().first()


async def get_weight_for_age_data(db: AsyncSession, age_years: int, gender: str):
    result = await db.execute(
        select(WeightForAge).filter(
            WeightForAge.year == age_years,
            WeightForAge.gender == gender
        )
    )
    return result.scalars().first()


async def predict_stunting(db: AsyncSession, child: Child):
    age_months = child.age
    height_for_age = await get_height_for_age_data(db, age_months, child.gender)

    if height_for_age:
        if child.height < height_for_age.sd_2_negative:
            return True
    return False


async def predict_wasting(db: AsyncSession, child: Child):
    age_months = child.age
    weight_for_age = await get_weight_for_age_data(db, age_months, child.gender)

    if weight_for_age:
        if child.weight < weight_for_age.sd_2_negative:
            return True
    return False


async def predict_overweight(db: AsyncSession, child: Child):
    age_months = child.age
    weight_for_age = await get_weight_for_age_data(db, age_months, child.gender)

    if weight_for_age:
        if child.weight > weight_for_age.sd_2_positive:
            return True
    return False


async def predict_child_condition(db: AsyncSession, child):
    is_stunting = await predict_stunting(db, child)
    is_wasting = await predict_wasting(db, child)
    is_overweight = await predict_overweight(db, child)

    return {
        "is_stunting": is_stunting,
        "is_wasting": is_wasting,
        "is_overweight": is_overweight
    }
