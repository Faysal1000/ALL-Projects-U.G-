using System;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;
using LifeinnovirorMentalHealthConsultency.Functional_Class;

namespace LifeinnovirorMentalHealthConsultency.Controllers.DoctorControllers
{
    public class DoctorTimeSlotManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;
        public DoctorTimeSlotManagementController()
        {
            db = new LifeinnovirorContext();
        }


        [Authorize(Roles = "Doctor")]
        [HttpPost]
        [Route("api/doctor/addTimeSlot")]
        public async Task<IHttpActionResult> AddTimeSlot(DoctorTimeSlot timeSlot)
        {
            try
            {
                // Get doctor ID from token
                int doctorId = CustomFunctions.GetDoctorUserIdFromToken(User);
                if (doctorId <= 0)
                {
                    return Content(HttpStatusCode.Unauthorized, new
                    {
                        success = false,
                        message = "Invalid doctor token."
                    });
                }

                // Validate date
                DateTime slotStartDateTime = timeSlot.Date.Date + timeSlot.StartTime;
                DateTime now = DateTime.Now;

                if (slotStartDateTime <= now)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Cannot add a slot in the past."
                    });
                }

                if ((timeSlot.Date - now.Date).TotalDays > CustomVariables.maxDaysInFutureDoctorCanAddTimeSlot)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = $"Cannot add a slot beyond {CustomVariables.maxDaysInFutureDoctorCanAddTimeSlot} days from today."
                    });
                }

                // Validate time
                if (timeSlot.EndTime <= timeSlot.StartTime)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "End time must be after start time."
                    });
                }


                // Validate session duration
                if ((timeSlot.EndTime - timeSlot.StartTime).TotalMinutes < CustomVariables.minDurationOfADoctorTimeSlotInMinutes)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = $"A session must be at least {CustomVariables.minDurationOfADoctorTimeSlotInMinutes} minutes long."
                    });
                }


                // Prevent overlapping slots for the same doctor on the same date
                bool isOverlapping = await db.DoctorTimeSlots.AnyAsync(slot =>
                    slot.DoctorId == doctorId &&
                    DbFunctions.TruncateTime(slot.Date) == timeSlot.Date.Date &&
                    timeSlot.StartTime < slot.EndTime &&
                    timeSlot.EndTime > slot.StartTime
                );

                if (isOverlapping)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "This time slot overlaps with an existing booked or available slot."
                    });
                }

               
                // Assign doctorId from token (prevent spoofing)
                timeSlot.DoctorId = doctorId;
                timeSlot.IsBooked = false;

                db.DoctorTimeSlots.Add(timeSlot);

                // Log: 
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Doctor",
                    ActorId = doctorId,
                    Action = "Add Timeslot Login",
                    Details = $"Doctor, Id='{doctorId}', Added a time slot, Id={timeSlot.SlotId}.",
                    CreatedAt = DateTime.Now
                });

                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Time slot added successfully.",
                    data = timeSlot
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while adding time slot.",
                    error = ex.Message
                });
            }
        }



        [Authorize(Roles = "Doctor")]
        [HttpPut]
        [Route("api/doctor/updateTimeSlot")]
        public async Task<IHttpActionResult> UpdateTimeSlot(DoctorTimeSlot updatedSlot)
        {
            try
            {
                // Get doctor ID from token
                int doctorId = CustomFunctions.GetDoctorUserIdFromToken(User);
                if (doctorId <= 0)
                {
                    return Content(HttpStatusCode.Unauthorized, new
                    {
                        success = false,
                        message = "Invalid doctor token."
                    });
                }

                var existingSlot = await db.DoctorTimeSlots.FindAsync(updatedSlot.SlotId);
                if (existingSlot == null || existingSlot.DoctorId != doctorId)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Time slot not found or not owned by this doctor."
                    });
                }

                if (existingSlot.IsBooked)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Cannot update a booked time slot."
                    });
                }

                DateTime slotStartDateTime = updatedSlot.Date.Date + updatedSlot.StartTime;
                DateTime now = DateTime.Now;

                if (slotStartDateTime <= now)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Cannot add a slot in the past."
                    });
                }

                if ((updatedSlot.Date - now.Date).TotalDays > CustomVariables.maxDaysInFutureDoctorCanAddTimeSlot)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = $"Cannot update a slot beyond {CustomVariables.maxDaysInFutureDoctorCanAddTimeSlot} days from today."
                    });
                }

                if (updatedSlot.EndTime <= updatedSlot.StartTime)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "End time must be after start time."
                    });
                }

                if ((updatedSlot.EndTime - updatedSlot.StartTime).TotalMinutes < CustomVariables.minDurationOfADoctorTimeSlotInMinutes)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = $"A session must be at least {CustomVariables.minDurationOfADoctorTimeSlotInMinutes} minutes long."
                    });
                }

                // Overlap check excluding current slot
                bool isOverlapping = await db.DoctorTimeSlots.AnyAsync(slot =>
                    slot.DoctorId == doctorId &&
                    slot.SlotId != updatedSlot.SlotId &&
                    DbFunctions.TruncateTime(slot.Date) == updatedSlot.Date.Date &&
                    updatedSlot.StartTime < slot.EndTime &&
                    updatedSlot.EndTime > slot.StartTime 
                );

                if (isOverlapping)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "This time slot overlaps with an existing slot."
                    });
                }

                // Update fields
                existingSlot.Date = updatedSlot.Date;
                existingSlot.StartTime = updatedSlot.StartTime;
                existingSlot.EndTime = updatedSlot.EndTime;

                // Logging
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Doctor",
                    ActorId = doctorId,
                    Action = "Update Timeslot",
                    Details = $"Doctor Id='{doctorId}' updated time slot Id={updatedSlot.SlotId}.",
                    CreatedAt = DateTime.Now
                });

                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Time slot updated successfully.",
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while updating time slot.",
                    error = ex.Message
                });
            }
        }




        [Authorize(Roles = "Doctor")]
        [HttpDelete]
        [Route("api/doctor/deleteTimeSlot/{id}")]
        public async Task<IHttpActionResult> DeleteTimeSlot(int id)
        {
            try
            {
                int doctorId = CustomFunctions.GetDoctorUserIdFromToken(User);
                if (doctorId <= 0)
                {
                    return Content(HttpStatusCode.Unauthorized, new
                    {
                        success = false,
                        message = "Invalid doctor token."
                    });
                }

                // Find the slot
                var slot = await db.DoctorTimeSlots.FirstOrDefaultAsync(s => s.SlotId == id);

                if (slot == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Time slot not found."
                    });
                }

                // Check ownership
                if (slot.DoctorId != doctorId)
                {
                    return Content(HttpStatusCode.Forbidden, new
                    {
                        success = false,
                        message = "You can only delete your own time slots."
                    });
                }

                // Check if already booked
                if (slot.IsBooked)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Cannot delete a slot that is already booked."
                    });
                }

                db.DoctorTimeSlots.Remove(slot);

                // Log:
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Doctor",
                    ActorId = doctorId,
                    Action = "Delete TimeSlot",
                    Details = $"Doctor, Id='{doctorId}', deleted a time slot, Id={slot.SlotId}.",
                    CreatedAt = DateTime.Now
                });

                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Time slot deleted successfully."
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "Unexpected error while deleting time slot.",
                    error = ex.Message
                });
            }
        }



        // this will get that particular doctor time slots
        [HttpGet]
        [Route("api/doctor/getTimeSlots")]
        public async Task<IHttpActionResult> GetDoctorTimeSlots()
        {
            try
            {
                // Get doctor ID from token
                int doctorId = CustomFunctions.GetDoctorUserIdFromToken(User);
                if (doctorId <= 0)
                {
                    return Content(HttpStatusCode.Unauthorized, new
                    {
                        success = false,
                        message = "Invalid doctor token."
                    });
                }

                //only show latest slots
                var now = DateTime.Now;
                var slots = await db.DoctorTimeSlots
                    .Where(s => s.DoctorId == doctorId &&
                                (DbFunctions.TruncateTime(s.Date) > now.Date ||
                                (DbFunctions.TruncateTime(s.Date) == now.Date && s.StartTime > now.TimeOfDay)))
                    .OrderBy(s => s.Date)
                    .ThenBy(s => s.StartTime)
                    .Select(s => new
                    {
                        s.SlotId,
                        s.Date,
                        StartTime = s.StartTime,
                        EndTime = s.EndTime,
                        s.IsBooked
                    })
                    .ToListAsync();

                if (slots == null || slots.Count == 0)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "No time slots found for the specified doctor."
                    });
                }

                return Ok(new
                {
                    success = true,
                    message = "Doctor time slots retrieved successfully.",
                    data = slots
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retrieving doctor time slots.",
                    error = ex.Message
                });
            }
        }




        // this can be accessible by any one. need to access in booking time
        [HttpGet]
        [Route("api/doctor/getTimeSlots/{id:int}")]
        public async Task<IHttpActionResult> GetDoctorTimeSlots(int id)
        {
            try
            {
                var now = DateTime.Now;
                var slots = await db.DoctorTimeSlots
                    .Where(s => s.DoctorId == id &&
                                (DbFunctions.TruncateTime(s.Date) > now.Date ||
                                (DbFunctions.TruncateTime(s.Date) == now.Date && s.StartTime > now.TimeOfDay)))
                    .OrderBy(s => s.Date)
                    .ThenBy(s => s.StartTime)
                    .Select(s => new
                    {
                        s.SlotId,
                        s.Date,
                        StartTime = s.StartTime,
                        EndTime = s.EndTime,
                        s.IsBooked
                    })
                    .ToListAsync();


                if (slots == null || slots.Count == 0)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "No time slots found for the specified doctor."
                    });
                }

                return Ok(new
                {
                    success = true,
                    message = "Doctor time slots retrieved successfully.",
                    data = slots
                });
            }
            catch (Exception ex)
            {
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An error occurred while retrieving doctor time slots.",
                    error = ex.Message
                });
            }
        }
    }
}
